from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HaMeR as a package',
    name='hamer',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        # 'opencv-python',
        'pyrender',
        'pytorch-lightning',
        # 'scikit-image',
        'smplx==0.1.28',
        # 'torch',
        # 'torchvision',
        'yacs',
        # 
        'chumpy @ git+https://github.com/mattloper/chumpy',
        'mmcv==1.3.9',
        'mmpose==0.29.0',
        # 'torch==2.1.0', # this can change depending on your cuda version [but in general, local cu12x works with other virtual env cu12y]
        # 'torch==2.1.0+cu121', # this can change depending on your cuda version [but in general, local cu12x works with other virtual env cu12y]
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
            'webdataset',
        ],
    },
)
