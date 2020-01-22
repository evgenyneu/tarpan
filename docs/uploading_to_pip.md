# Uploading this package to pip


1. Increment version in [setup.py](setup.py).

2. Delete dist directory and run:

```
python3 setup.py sdist bdist_wheel
```

3. Upload

```
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

4. Uninstall tarpan


```
pip uninstall tarpan-evgenyneu
```

5. Install tarpan

```
pip install --index-url https://test.pypi.org/simple/ --no-deps tarpan-evgenyneu
```
