import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "TDY_PKG"
USER_NAME = "saquibquddus"

setuptools.setup(
    name=f"{PROJECT_NAME}-{USER_NAME}",
    version="1.1.1",
    author=USER_NAME,
    author_email="sakibquddus@gmail.com",
    description="its an implimentation of TF-2 , Detectron and yolov5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=
    [
    "seaborn",
    "beautifulsoup4==4.11.1",
    "filelock==3.6.0",
    "gdown==4.4.0",
    "opencv-contrib-python==4.1.2.30",
    "PySocks==1.7.1",
    "PyYAML==6.0",
    "scipy==1.7.3",
    "soupsieve==2.3.2",
    "tqdm==4.64.0",  
    "typing_extensions",
    "flask",
    "flask-cors",
    "gunicorn",
    "torch",
    "torchvision",
    "torchaudio",
    "pycocotools",
    "opencv-python-headless==4.5.2.52",
    "fvcore",
    "omegaconf",
    "pandas",
    "absl-py",
    "apache-beam",
    "astunparse",
    "avro-python3",
    "Bottleneck",
    "cached-property",
    "cachetools",
    "certifi",
    "charset-normalizer",
    "click",
    "cloudpickle",
    "colorama",
    "crcmod",
    "cycler",
    "Cython",
    "dill",
    "docopt",
    "fastavro",
    "flatbuffers",
    "fonttools",
    "gast",
    "google-auth",
    "google-auth-oauthlib",
    "google-pasta",
    "grpcio",
    "h5py",
    "hdfs",
    "httplib2",
    "idna",
    "importlib-metadata",
    "itsdangerous",
    "Jinja2",
    "keras-nightly",
    "Keras-Preprocessing",
    "kiwisolver",
    "Markdown",
    "MarkupSafe",
    "matplotlib",
    "mkl-fft",
    "mkl-service",
    "munkres",
    "numpy",
    "oauth2client",
    "oauthlib",
    "opencv-python",
    "opt-einsum",
    "orjson",
    "packaging",
    "Pillow",
    "proto-plus",
    "protobuf",
    "pyarrow",
    "pyasn1",
    "pyasn1-modules",
    "pydot",
    "pymongo",
    "pyparsing",
    "python-dateutil",
    "pytz",
    "requests",
    "requests-oauthlib",
    "rsa",
    "sip",
    "six",
    "tensorboard==2.5",
    "tensorflow==2.5",
    "tensorflow-estimator==2.5.0",
    "termcolor",
    "tf-models-official",
    "tf-slim",
    "typing-extensions",
    "urllib3",
    "Werkzeug",
    "wincertstore",
    "wrapt",
    "zipp"
    ]
)