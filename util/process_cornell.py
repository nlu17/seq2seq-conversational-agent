DATA_DIR = "../data/cornell_movie_dialogs_corpus/"
LINES_FILE = "processed_lines.txt"
TEMP_FILE = "temp_conversations.txt"
OUT_FILE = "processed_conversations.txt"
NAMES_FILE = "../data/names.txt"

names = {}
lines = {}

with open(NAMES_FILE, "r") as f:
    names_list = [line.strip() for line in f.readlines()]
    names = dict(zip(names_list, [0] * len(names_list)))

with open(DATA_DIR+LINES_FILE, "r", encoding='windows-1252') as f:
    raw_lines = [line.strip().split("\t") for line in f.readlines()]
    for data in raw_lines:
        if len(data) < 2:
            continue

        # Replace names with <person>.
        line = data[1].split()
        line = map(lambda x: x.lower() if x not in names else "<person>", line)
        line = " ".join(line)

        lines[data[0]] = line

with open(DATA_DIR+TEMP_FILE, "r") as fin, \
        open(DATA_DIR+OUT_FILE, "w", encoding="utf-8") as fout:
    # Make triplets.
    raw_convos = [line.strip("\n[]") for line in fin.readlines()]
    for convo in raw_convos:
        convo = convo.split()
        convo = list(map(
            lambda line_id: lines[line_id] if line_id in lines else None,
            convo))

        if None in convo:
            continue

        convo = "\t".join(convo)
        fout.write(convo + "\n")
