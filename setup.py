from setuptools import setup

setup(
    name="movie-recommender-system",
    version="0.0.1",
    author="Pradyot Srivastava",
    author_email="pradyotsrivastava24@gmail.com",
    description="A Movie Recommendation System using Machine Learning",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "streamlit"
    ],
)
