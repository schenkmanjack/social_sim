from setuptools import setup, find_packages

setup(
    name='social_gym',
    version='0.1.0',
    description='Codebase for evolutionary optimization of social agents',
    author='Jack Schenkman',
    author_email='schenkmanjack@gmail.com',
    packages=find_packages(),
    install_requires=[
        'openai',
        'python-dotenv',
    ],
    python_requires='>=3.8',
)
