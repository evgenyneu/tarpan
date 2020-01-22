# Uploading this package to pip


1. Delete dist directory and run:

```
python3 setup.py sdist bdist_wheel
```

2. Upload

```
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

3. Install

```
pip install --index-url https://test.pypi.org/simple/ --no-deps tarpan-evgenyneu
```
