from setuptools import setup, find_packages

setup(
    name="mito-linescan",
    version="0.1",
    description="Mitochondrial protein line scanner tools",
    author="Hamid Rahmani, Michaela Horger",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mito_mask=bin.mito_mask:main",
            "mito_mask_refine=bin.mito_mask_refine:main",
            "mito_protein_line_scanner=bin.mito_protein_line_scanner:main",
            "mito_protein_omm_localization=bin.mito_protein_omm_localization:main",
        ],
    },
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-image",
        "networkx",
        "tqdm",
        "click",
        "tifffile",
        "sknw",
    ],
)
