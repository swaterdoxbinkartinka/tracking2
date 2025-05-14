from setuptools import setup, find_packages

setup(
    name='yolo-tracker',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'ultralytics>=8.0.0',
        'opencv-python',
        'deep_sort_realtime',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'yolo-tracker=cli:main',
        ],
    },
    author='invokegpt',
    description='YOLO + DeepSORT трекинг',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8',
)
