import subprocess

# List of packages your project actually uses
needed = {
    "streamlit",
    "pandas",
    "numpy",
    "nltk",
    "seaborn",
    "matplotlib",
    "scikit-learn",
    "imbalanced-learn"
}

# Run pip freeze
installed = subprocess.check_output(["pip", "freeze"], text=True).splitlines()

# Filter only needed packages
filtered = [pkg for pkg in installed if pkg.split("==")[0].lower() in needed]

# Write to requirements.txt
with open("requirements_1.txt", "w") as f:
    f.write("\n".join(filtered))

print("Clean requirements.txt generated!")
