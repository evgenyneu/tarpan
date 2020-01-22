## Uploading this package to [pypi.org](https://pypi.org)

Tutorial: https://packaging.python.org/tutorials/packaging-projects/

1. Increment version in [setup.py](setup.py).

2. Delete dist directory and run:

```
python3 setup.py sdist bdist_wheel
```

3. Upload

```
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
```

4. Uninstall tarpan


```
pip uninstall tarpan
```

5. Install tarpan

```
pip install tarpan
```

## Uploading this package to test pip


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
