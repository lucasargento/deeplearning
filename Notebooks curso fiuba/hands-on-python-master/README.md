# hands-on-python

El objetivo de este módulo es brindar material de nivelación en Python y librerías Numpy y Matplotlib.
Además, se recomienda el curso de Python en Kaggle: (https://www.kaggle.com/learn/python)

##### Requirements
* Python 3
* Pip 3

En caso de no tener los requerimientos instalados realizar los siguientes pasos, de acuerdo al SO que poseas:

## macOS

Instalar Python 3
```bash
$ brew install python3
```

pip3 ya viene incluido, por lo que no hace falta instalarlo

##### Installation de virtualenv
Para instalar `virtualenv` via pip ejecutar:
```bash
$ pip3 install virtualenv
```

##### Uso de virtualenv
Creación de un virtualenv:
```bash
$ virtualenv -p python3 <desired-path>/<name_env>
```

Activar el entorno virtual:
```bash
$ source <desired-path>/<name_env>/bin/activate
```

Instalar requerimientos:
```bash
$ pip3 install -r requirements.txt
```

Desactivar el virtualenv:
```bash
$ deactivate
```

## Ubuntu 18.04 o superior

Instalar Python 3
```bash
$ sudo apt-get update
$ sudo apt-get install python3.9 python3-pip python-virtualenv virtualenv
```

Crear entorno virtual
```bash
$ virtualenv <desired-path>/<name_env> --python=python3
```

Activar el entorno virtual:
```bash
$ source <desired-path>/<name_env>/bin/activate
```

Instalar requerimientos:
```bash
$ pip3 install -r requirements.txt
```

Desactivar el virtualenv:
```bash
$ deactivate
```


## Windows 10 (para otras versiones debería funcionar)

Descargamos e instalamos Python 3 para Windows
https://www.python.org/downloads/

Verificar que pip se haya instalado correctamente
```bash
$ pip --version
```

Instalar Virtualenv
```bash
$ pip install virtualenv
```

Crear un entorno virtual
```bash
$ virtualenv <desired-path>/<name_env>
```

Activar el entorno virtual
```bash
$ virtualenv <desired-path>/<name_env>/activate
```

Instalar requerimientos:
```bash
$ pip3 install -r requirements.txt
```

Desactivar el virtualenv:
```bash
$ deactivate
```

## Otras propuestas para utilizar

**ANACONDA Suite** (https://www.anaconda.com/products/individual)

Es una distribución de Python que básicamente funciona como un gestor de entorno virtual y de paquetes.

**Google Colab** (https://colab.research.google.com/)

Te permite ejecutar y programar en Python en un entorno web, similar a un Jupyter Notebook convencional. Su principal ventaja es que no se requiere ningún tipo de configuración, provee acceso a uso de GPU de forma gratuita y permite compartir código muy fácilmente.

**Contenedores**

Un contenedor es un ambiente de ejecución que provee a una aplicación lo básico para funcionar. A diferencia de las máquinas virtuales, los contenedores utilizan el SO de host en lugar de uno propio.

Se provee un Dockerfile listo para crear un contenedor.
(https://github.com/EGRIS-DeepLearning/docker-notebook)

