#!/bin/bash

cd /opt
git clone https://github.com/labsyspharm/minerva-lib-python.git

/opt/python/cp36-cp36m/bin/pip wheel /opt/minerva-lib-python -w /wheels
/opt/python/cp37-cp37m/bin/pip wheel /opt/minerva-lib-python -w /wheels
/opt/python/cp38-cp38/bin/pip wheel /opt/minerva-lib-python -w /wheels
#/opt/python/cp39-cp39/bin/pip wheel /opt/minerva-lib-python -w /opt/minerva-lib-python/output

find /wheels -name "minerva_lib*whl" | xargs -I % auditwheel repair % -w /wheels