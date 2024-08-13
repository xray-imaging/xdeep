from setuptools import setup, Extension, find_packages

setup(
    name = 'xfusion',
    author = 'Songyuan Tang',
    author_email = 'tangs@anl.gov',
    description = 'add ...',
    packages = find_packages(),
    entry_points={'console_scripts':['xfusion = xfusion.__main__:main'],},
    version = open('VERSION').read().strip(),
    zip_safe = False,
    url='http://xfusion.readthedocs.org',
    download_url='https://github.com/decarlof/xfusion.git',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
)

