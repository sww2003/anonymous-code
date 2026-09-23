#!/usr/bin/env python

import os

from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    with open('VERSION', encoding='utf-8') as file:
        return file.read().strip()


def make_cuda_ext(name, module, sources, sources_cuda=None):
    if sources_cuda is None:
        sources_cuda = []
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


if __name__ == '__main__':
    cuda_ext = os.getenv('BASICSR_EXT')  # whether compile cuda ext
    if cuda_ext == 'True':
        try:
            import torch
            from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
        except ImportError:
            raise ImportError('Unable to import torch - torch is needed to build cuda extensions')

        ext_modules = [
            make_cuda_ext(
                name='deform_conv_ext',
                module='basicsr.ops.dcn',
                sources=['src/deform_conv_ext.cpp'],
                sources_cuda=['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu']),
            make_cuda_ext(
                name='fused_act_ext',
                module='basicsr.ops.fused_act',
                sources=['src/fused_bias_act.cpp'],
                sources_cuda=['src/fused_bias_act_kernel.cu']),
            make_cuda_ext(
                name='upfirdn2d_ext',
                module='basicsr.ops.upfirdn2d',
                sources=['src/upfirdn2d.cpp'],
                sources_cuda=['src/upfirdn2d_kernel.cu']),
        ]
        setup_kwargs = dict(cmdclass={'build_ext': BuildExtension})
    else:
        ext_modules = []
        setup_kwargs = dict()

    setup(
        name='phoenixsr',
        version=get_version(),
        description=(
            'PhoenixSR: Generative Heterogeneous Distillation Unleashes '
            'Efficient Models for Real-World Super-Resolution'),
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='Anonymous',
        python_requires='>=3.10',
        keywords='computer vision, image restoration, super resolution, distillation',
        include_package_data=True,
        packages=find_packages(exclude=('options', 'datasets', 'experiments', 'results', 'tb_logger', 'wandb')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
        ],
        license='Apache License 2.0',
        install_requires=get_requirements(),
        ext_modules=ext_modules,
        zip_safe=False,
        **setup_kwargs)
