from setuptools import setup, find_packages

setup(
    name='ada-guardian',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'llama-cpp-python>=0.2.0',
        'faster-whisper>=0.10.0',
    ],
    entry_points={
        'console_scripts': [
            'ada-guardian=ada_guardian.cli:main',
        ],
    },
    author='Ada Guardian Team',
    description='AI-powered guardian system with local LLM and speech recognition',
    python_requires='>=3.8',
)
