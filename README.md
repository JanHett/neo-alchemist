# Neo Alchemist

Utility for editing digitised colour negatives.

## Setup

Most dependencies can be installed in a pipenv:

```sh
pipenv install
```

The only exception is OpenColorIO 2.0.x which needs to be installed globally:

```sh
# macOS
brew install opencolorio

# RHEL-like Linux
yum install OpenColorIO
```

You also need to source OCIO's config file, e.g. for macOS

```sh
source $(brew --prefix)/share/ocio/setup_ocio.sh
```

Finally, you need to get the OCIO configs. These are included as a submodule, so you can pull them with

```sh
git submodule update --init --recursive
```

Then you can point OCIO at the relevant config by setting the `OCIO` environment variable to the path of the spi-vfx config.
