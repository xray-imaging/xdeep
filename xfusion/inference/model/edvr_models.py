import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from ops.dcn.deform_conv import ModulatedDeformConvPack

import logging
from distutils.version import LooseVersion

initialized_logger = {}

class MultiHeadedAttention_LoHi(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model, scale):
        super().__init__()

        self.patchsize = patchsize
        self.d_model = d_model
        self.query_embedding_hi = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding_hi = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding_hi = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear_hi = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.output_linear_lo = nn.Sequential(
            nn.Conv2d(d_model*scale**2, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

        assert type(scale) is int
        self.scale = scale

        self.query_embedding_lo = nn.Conv2d(d_model, d_model * scale**2, kernel_size=1, padding=0)
        self.value_embedding_lo = nn.Conv2d(d_model, d_model * scale**2, kernel_size=1, padding=0)
        self.key_embedding_lo = nn.Conv2d(d_model, d_model * scale**2, kernel_size=1, padding=0)

    def forward(self, x_lo, x_hi, b):
        bt_lo, _, h, w = x_lo.size()
        t_lo = bt_lo // b
        d_k_lo = self.d_model * self.scale**2 // len(self.patchsize)

        bt_hi, _, h_, w_ = x_hi.size()
        t_hi = bt_hi // b
        d_k_hi = self.d_model // len(self.patchsize)

        assert h_ == h * self.scale
        assert w_ == w * self.scale
        output_lo, output_hi = [], []

        _query_lo = self.query_embedding_lo(x_lo)
        _key_lo = self.key_embedding_lo(x_lo)
        _value_lo = self.value_embedding_lo(x_lo)
        _query_hi = self.query_embedding_hi(x_hi)
        _key_hi = self.key_embedding_hi(x_hi)
        _value_hi = self.value_embedding_hi(x_hi)

        for (width, height), query_lo, key_lo, value_lo, query_hi, key_hi, value_hi in zip(self.patchsize,
                                                      torch.chunk(_query_lo, len(self.patchsize), dim=1),
                                                      torch.chunk(_key_lo, len(self.patchsize), dim=1),
                                                      torch.chunk(_value_lo, len(self.patchsize), dim=1),
                                                      torch.chunk(_query_hi, len(self.patchsize), dim=1),
                                                      torch.chunk(_key_hi, len(self.patchsize), dim=1),
                                                      torch.chunk(_value_hi, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height

            # 1) embedding and reshape
            query_lo = query_lo.view(b, t_lo, d_k_lo, out_h, height, out_w, width)
            query_lo = query_lo.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t_lo*out_h*out_w, d_k_lo*height*width)
            key_lo = key_lo.view(b, t_lo, d_k_lo, out_h, height, out_w, width)
            key_lo = key_lo.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t_lo*out_h*out_w, d_k_lo*height*width)
            value_lo = value_lo.view(b, t_lo, d_k_lo, out_h, height, out_w, width)
            value_lo = value_lo.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t_lo*out_h*out_w, d_k_lo*height*width)

            query_hi = query_hi.view(b, t_hi, d_k_hi, out_h, height*self.scale, out_w, width*self.scale)
            query_hi = query_hi.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t_hi*out_h*out_w, d_k_hi*height*width*(self.scale)**2)
            key_hi = key_hi.view(b, t_hi, d_k_hi, out_h, height*self.scale, out_w, width*self.scale)
            key_hi = key_hi.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t_hi*out_h*out_w, d_k_hi*height*width*(self.scale)**2)
            value_hi = value_hi.view(b, t_hi, d_k_hi, out_h, height*self.scale, out_w, width*self.scale)
            value_hi = value_hi.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t_hi*out_h*out_w, d_k_hi*height*width*(self.scale)**2)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            query = torch.cat([query_lo,query_hi],axis=1)
            key = torch.cat([key_lo,key_hi],axis=1)
            value = torch.cat([value_lo,value_hi],axis=1)
            y, _ = self.attention(query, key, value, None)
            # 3) "Concat" using a view and apply a final linear.
            y_lo = y[:,:t_lo*out_h*out_w,:].view(b, t_lo, out_h, out_w, d_k_lo, height, width)
            y_hi = y[:,t_lo*out_h*out_w:,:].view(b, t_hi, out_h, out_w, d_k_hi, height*self.scale, width*self.scale)
            #y = y.view(b, t, out_h, out_w, d_k, height, width)
            #y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            y_lo = y_lo.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt_lo, d_k_lo, h, w)
            y_hi = y_hi.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt_hi, d_k_hi, h_, w_)
            output_lo.append(y_lo)
            output_hi.append(y_hi)
        output_lo = torch.cat(output_lo, 1)
        output_hi = torch.cat(output_hi, 1)
        x_lo = self.output_linear_lo(output_lo)
        x_hi = self.output_linear_hi(output_hi)
        return x_lo, x_hi

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, m, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        attn = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), torch.chunk(
                                                          _key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            if m is not None:
                mm = m.view(b, t, 1, out_h, height, out_w, width)
                mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                    b,  t*out_h*out_w, height*width)
                mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            else:
                mm = None
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, p_attn = self.attention(query, key, value, mm)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
            attn.append(p_attn)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x, attn

class TransformerBlock_LoHi(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128, scale = 4):
        super().__init__()
        self.attention = MultiHeadedAttention_LoHi(patchsize, d_model=hidden, scale = scale)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c = x['x'], x['m'], x['b'], x['c']
        x_lo, x_hi = torch.clone(x[0]), torch.clone(x[1])
        x_lo_att, x_hi_att = self.attention(x_lo, x_hi, b)
        x_lo = x_lo + x_lo_att
        x_hi = x_hi + x_hi_att
        x_lo = x_lo + self.feed_forward(x_lo)
        x_hi = x_hi + self.feed_forward(x_hi)
        return {'x': (x_lo, x_hi), 'm': m, 'b': b, 'c': c}

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c, a = torch.clone(x['x']), x['m'], x['b'], x['c'], x['attn']
        
        x_attn, attn = self.attention(x, m, b, c)
        a.extend(attn)
        x = x + x_attn
        x = x + self.feed_forward(x)
        return {'x': x, 'm': m, 'b': b, 'c': c, 'attn': a}

class EDVRSTF(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 num_frame_hi = 1,
                 center_frame_idx=None,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True,
                 with_transformer = False,
                 patchsize = None,
                 stack_num = None,
                 fuse_searched_feat_ok = False,
                 num_frame_search = None,
                 downsample_hi_ok = True,
                 num_hidden_feat = None,
                 scale = None):
        super(EDVRSTF, self).__init__()
        assert center_frame_idx is not None
        if num_frame_hi is None:
            num_frame_hi = 1
        self.center_frame_idx = center_frame_idx
        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa
        self.with_transformer = with_transformer
        self.downsample_hi_ok = downsample_hi_ok
        if not downsample_hi_ok:
            assert type(num_hidden_feat) is int
            assert type(scale) is int
        #self.num_in_ch = num_in_ch
        #self.num_out_ch = num_out_ch
        #self.num_feat = num_feat
        #self.num_frame = num_frame
        #self.deformable_groups = deformable_groups
        #self.num_extract_block = num_extract_block
        #self.num_reconstruct_block = num_reconstruct_block
        
        # extract features for each frame
        #if self.with_predeblur:
        #    self.predeblur = PredeblurModule(num_feat=num_feat, hr_in=self.hr_in)
        #    self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        #else:
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extract features for high resolution frame
        self.conv_first_hi = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1_hi = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2_hi = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.max_pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        # extract pyramid features
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        if self.with_tsa:
            self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame+num_frame_hi, center_frame_idx=self.center_frame_idx,\
                                    fuse_searched_feat_ok=fuse_searched_feat_ok, num_frame_search=num_frame_search)
        else:
            self.fusion = nn.Conv2d((num_frame+num_frame_hi) * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if with_transformer:
            blocks = []
            for _ in range(stack_num):
                if self.downsample_hi_ok:
                    blocks.append(TransformerBlock(patchsize, hidden=num_feat))
                else:
                    blocks.append(TransformerBlock_LoHi(patchsize, hidden=num_hidden_feat, scale=scale))
                    self.transformer_conv_in = nn.Conv2d(num_feat,num_hidden_feat,1,1)
                    self.transformer_conv_out = nn.Conv2d(num_hidden_feat,num_feat,1,1)
            self.transformer = nn.Sequential(*blocks)

            if not fuse_searched_feat_ok:
                self.post_fusion_ = nn.Conv2d(2 * num_feat, num_feat, 1, 1)

    def unfreeze_model(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, sample):
        x, y = sample['lq'], sample['hq']
        b, t, c, h, w = x.size()
        b_,t_,c_,h_,w_ = y.size()

        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        if self.with_predeblur:
            feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
            if self.hr_in:
                h, w = h // 4, w // 4
        else:
            feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        y = self.lrelu(self.conv_first_hi(y))
        if not self.downsample_hi_ok:
            if self.with_transformer:
                feat_l1_in = self.transformer_conv_in(feat_l1)
                y_in = self.transformer_conv_in(y)
                feat_attention = self.transformer({'x':(feat_l1_in,y_in),'m':None,'b':b,'c':feat_l1.size()[1]})['x'][0].view(b, t, -1, h, w)
                feat_attention_center = feat_attention[:, self.center_frame_idx, :,:,:]
                feat_attention_center = self.transformer_conv_out(feat_attention_center)
        
        y = self.max_pool1(self.lrelu(self.conv1_hi(y)))
        y = self.max_pool2(self.lrelu(self.conv2_hi(y)))
        feat_l1 = torch.cat((feat_l1.view(b, t, -1, h, w), y.view(b, 1, -1, h, w)), dim=1)
        t = t + 1
        feat_l1 = feat_l1.view(b * t, -1, h, w)

        if self.downsample_hi_ok:
            if self.with_transformer:
                feat_attention = self.transformer({'x':feat_l1,'m':None,'b':b,'c':feat_l1.size()[1], 'attn': []})['x'].view(b, t, -1, h, w)
                feat_attention_center = feat_attention[:, self.center_frame_idx, :,:,:]

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        #feat = self.fusion(aligned_feat)
        if self.fusion.fuse_searched_feat_ok and self.with_transformer:
            feat = self.fusion((aligned_feat, feat_attention_center.unsqueeze(1)))
        else:
            feat = self.fusion(aligned_feat)
        
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        
        return out


class EDVRSTFTempRank(EDVRSTF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_in_ch, num_out_ch, num_feat, num_frame, deformable_groups, num_extract_block, num_reconstruct_block,\
            center_frame_idx, with_predeblur = kwargs['num_in_ch'], kwargs['num_out_ch'], kwargs['num_feat'],kwargs['num_frame'],kwargs['deformable_groups'],\
            kwargs['num_extract_block'],kwargs['num_reconstruct_block'],kwargs['center_frame_idx'],kwargs['with_predeblur']
        if 'with_transformer' in list(kwargs.keys()):
            with_transformer, patchsize, stack_num, fuse_searched_feat_ok, num_frame_search\
            = kwargs['with_transformer'],kwargs['patchsize'],kwargs['stack_num'],kwargs['fuse_searched_feat_ok'],kwargs['num_frame_search']
            if self.with_tsa:
                self.fusion = TSAFusionTempRank(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx,\
                                            fuse_searched_feat_ok=fuse_searched_feat_ok, num_frame_search=num_frame_search)
            else:
                self.fusion = nn.Conv2d(self.num_frame * self.num_feat, self.num_feat, 1, 1)
        else:
            if self.with_tsa:
                self.fusion = TSAFusionTempRank(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)
            else:
                self.fusion = nn.Conv2d(self.num_frame * self.num_feat, self.num_feat, 1, 1)

        if 'downsample_hi_ok' in list(kwargs.keys()):
            downsample_hi_ok, num_hidden_feat, scale = kwargs['downsample_hi_ok'],kwargs['num_hidden_feat'], kwargs['scale']
    
    def forward(self, sample):
        x, y = sample['lq'], sample['hq']
        b, t1, c, h, w = x.size()
        b_, t2, c_, h_, w_ = y.size()
        #assert (b, c, h, w) == (b_ ,c_, h_, w_)
        assert self.fusion.num_frame <= (t1+t2)
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, ('The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, ('The height and width must be multiple of 4.')

        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # extract features for each frame
        # L1
        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        y = self.lrelu(self.conv_first_hi(y.view(-1,c_,h_,w_)))
        if not self.downsample_hi_ok:
            if self.with_transformer:
                feat_l1_in = self.transformer_conv_in(feat_l1)
                y_in = self.transformer_conv_in(y)
                feat_attention = self.transformer({'x':(feat_l1_in,y_in),'m':None,'b':b,'c':feat_l1.size()[1]})['x'][0].view(b, t, -1, h, w)
                feat_attention_center = feat_attention[:, self.center_frame_idx, :,:,:]
                feat_attention_center = self.transformer_conv_out(feat_attention_center)

        y = self.max_pool1(self.lrelu(self.conv1_hi(y)))
        y = self.max_pool2(self.lrelu(self.conv2_hi(y)))
        feat_l1 = torch.cat((feat_l1.view(b, t1, -1, h, w), y.view(b, t2, -1, h, w)), dim=1)
        t = t1 + t2
        feat_l1 = feat_l1.view(b * t, -1, h, w)

        if self.downsample_hi_ok:
            if self.with_transformer:
                results_ = self.transformer({'x':feat_l1,'m':None,'b':b,'c':feat_l1.size()[1], 'attn': []})
                
                import numpy as np
                for i in range(32):
                    np.save(f'attn_ctc_50_{i}_282.npy', results_['attn'][i].squeeze().detach().cpu().numpy())
                
                feat_attention = results_['x'].view(b, t, -1, h, w)
                feat_attention_center = feat_attention[:, self.center_frame_idx, :,:,:]

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(), feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(), feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        #feat, corr_score = self.fusion(aligned_feat)
        if self.fusion.fuse_searched_feat_ok and self.with_transformer:
            feat, corr_score = self.fusion((aligned_feat, feat_attention_center.unsqueeze(1)))
        else:
            feat, corr_score = self.fusion(aligned_feat)
        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        results = {'out':out,'corr_score':corr_score,'aligned_feat':aligned_feat}
        return results
    
class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2,fuse_searched_feat_ok = False, num_frame_search = None):
        super(TSAFusion, self).__init__()
        self.num_feat = num_feat
        self.num_frame = num_frame
        self.center_frame_idx = center_frame_idx
        self.fuse_searched_feat_ok = fuse_searched_feat_ok
        if fuse_searched_feat_ok:
            assert num_frame_search is not None
            self.num_frame_search = num_frame_search
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        #self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        if fuse_searched_feat_ok:
            self.temporal_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if fuse_searched_feat_ok:
            self.feat_fusion = nn.Conv2d((num_frame + num_frame_search) * num_feat, num_feat, 1, 1)
        else:
            self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)
        
        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        #self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        if fuse_searched_feat_ok:
            self.spatial_attn1 = nn.Conv2d((num_frame + num_frame_search) * num_feat, num_feat, 1)
        else:
            self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, feats):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        if self.fuse_searched_feat_ok:
            aligned_feat, searched_feat = feats
        else:
            aligned_feat = feats

        b, t, c, h, w = aligned_feat.size()
        if self.fuse_searched_feat_ok:
            b_, t_, c_, h_, w_ = searched_feat.size()
            assert (b, c, h, w) == (b_, c_, h_, w_)
            assert t_ == self.num_frame_search
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)
        if self.fuse_searched_feat_ok:
            embedding_ = self.temporal_attn3(searched_feat.view(-1, c, h, w))
            embedding_ = embedding_.view(b, t_, -1, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        if self.fuse_searched_feat_ok:
            for i in range(t_):
                emb_neighbor = embedding_[:, i, :, :, :]
                corr = torch.sum(emb_neighbor * embedding_ref, 1) # (b, h, w)
                corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        if self.fuse_searched_feat_ok:
            corr_prob = corr_prob.unsqueeze(2).expand(b, (t+t_), c, h, w)
        else:
            corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        #corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        #aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
        if self.fuse_searched_feat_ok:
            aligned_feat = torch.cat([aligned_feat, searched_feat], dim=1).view(b, -1, h, w) * corr_prob
        else:
            aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat

class TSAFusionTempRank(TSAFusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, feats):

        if self.fuse_searched_feat_ok:
            aligned_feat, searched_feat = feats
        else:
            aligned_feat = feats

        b, t, c, h, w = aligned_feat.size()
        if self.fuse_searched_feat_ok:
            b_, t_, c_, h_, w_ = searched_feat.size()
            assert (b, c, h, w) == (b_, c_, h_, w_)
            assert t_ == self.num_frame_search
        
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)
        if self.fuse_searched_feat_ok:
            embedding_ = self.temporal_attn3(searched_feat.view(-1, c, h, w))
            embedding_ = embedding_.view(b, t_, -1, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        if self.fuse_searched_feat_ok:
            for i in range(t_):
                emb_neighbor = embedding_[:, i, :, :, :]
                corr = torch.sum(emb_neighbor * embedding_ref, 1) # (b, h, w)
                corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_score = self.avgpool(torch.cat(corr_l, dim=1)).squeeze().unsqueeze(0) # (b, t)
        
        if self.fuse_searched_feat_ok:
            if (t+t_) > (self.num_frame+t_):
                topk_times = torch.topk(corr_score, self.num_frame, dim=1)[1] # (b, k)
                # keep index of reference frame unchanged
                for i in range(b):
                    assert self.center_frame_idx in topk_times[i]
                    j = (topk_times[i]==self.center_frame_idx).nonzero(as_tuple=True)[0][0]#.numpy()
                    #temp = topk_times[i][j]
                    topk_times[i][j] = topk_times[i][self.center_frame_idx]
                    topk_times[i][self.center_frame_idx] = self.center_frame_idx
        else:
            if t > self.num_frame:
                topk_times = torch.topk(corr_score, self.num_frame, dim=1)[1] # (b, k)
                # keep index of reference frame unchanged
                for i in range(b):
                    assert self.center_frame_idx in topk_times[i]
                    j = (topk_times[i]==self.center_frame_idx).nonzero(as_tuple=True)[0][0]#.numpy()
                    #temp = topk_times[i][j]
                    topk_times[i][j] = topk_times[i][self.center_frame_idx]
                    topk_times[i][self.center_frame_idx] = self.center_frame_idx
        
        #corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        if self.fuse_searched_feat_ok:
            corr_prob = corr_prob.unsqueeze(2).expand(b, (t+t_), c, h, w)
            aligned_feat = torch.cat((aligned_feat,searched_feat),dim=1)
        else:
            corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        # keep the topk feature and corr maps along time dimension
        if self.fuse_searched_feat_ok:
            if (t+t_) > (self.num_frame+t_):
                corr_prob = corr_prob.gather(1,topk_times[:,:,None,None,None].expand([b,self.num_frame,c,h,w])) # (b, k, c, h, w)
                aligned_feat = aligned_feat.gather(1,topk_times[:,:,None,None,None].expand([b,self.num_frame,c,h,w])) # (b, k, c, h, w)
        else:
            if t > self.num_frame:
                corr_prob = corr_prob.gather(1,topk_times[:,:,None,None,None].expand([b,self.num_frame,c,h,w])) # (b, k, c, h, w)
                aligned_feat = aligned_feat.gather(1,topk_times[:,:,None,None,None].expand([b,self.num_frame,c,h,w])) # (b, k, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, k*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat, corr_score
    
class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    ``Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks``

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).

        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        assert LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0')
        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                            self.dilation, mask)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    import torch.distributed as dist
    def get_dist_info():
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
        if initialized:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    from torch.nn import init as init
    from torch.nn.modules.batchnorm import _BatchNorm

    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)