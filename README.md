# Neo Alchemist

Utility for editing digitised colour negatives.

## Setup

Most dependencies can be installed in a pipenv:

```sh
pipenv install
```

The only exceptions are OpenImageIO and OpenColorIO 2.0.x which need to be installed globally:

```sh
# macOS
brew install openimageio opencolorio

# RHEL-like Linux
yum install OpenImageIO OpenColorIO
```

You also need to source OCIO's config file, e.g. for macOS

```sh
source $(brew --prefix)/share/ocio/setup_ocio.sh
```

Depending on your system config, you might also have to add OIIO and OCIO to your `PYTHONPATH`.

Finally, you need to get the OCIO configs. These are included as a submodule, so you can pull them with

```sh
git submodule update --init --recursive
```

Then you can point OCIO at the relevant config by setting the `OCIO` environment variable to the path of the spi-vfx config.
