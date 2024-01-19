import os

fnames = os.listdir("cifs/")

fnames = sorted(fnames)

bash_command = "#!/usr/bin/env bash\n"


for fname in fnames:
    bash_command += f"./network -gridG zeolites/iza_cifs/{fname} &\n"

print(bash_command)