import re


if __name__ == "__main__":
    path = "/data/SplaTAM_cyh/log/03011537.txt"
    out_path = "/data/SplaTAM_cyh/number_of_gaussians_origin.txt"
    pattern = r"Number of gaussians in frame \d+: \d+"
    with open(path, "r") as f:
        data = f.read()
        result = re.findall(pattern, data)
        with open(out_path, "w") as out:
            for line in result:
                out.write(line[29:] + "\n")
                
        