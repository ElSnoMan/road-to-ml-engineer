# 1. Update pyenv
cd /home/gitpod/.pyenv/plugins/python-build/../.. && git pull && cd -

# 2. Install specific python version (default is 3.10.1)
if [ -z $1 ]
then
    pyenv install $1
    pyenv global $1
else
    pyenv install 3.10.1
    pyenv global 3.10.1
fi

# 3. Install poetry as package manager
pip install poetry
