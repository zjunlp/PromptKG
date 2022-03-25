
entity2type = {}
with open("dataset/FB15k-237/entity2type.txt", "r") as file:
    for line in file.readlines():
        line = line.strip().replace("\n", "")
        ids = line.split("\t")[0]
        entity_type = line.split("\t")[1].replace("/", " ").replace("_", " ")
        assert "/" in ids
        entity2type[ids] = " ".join(list(set(entity_type.split(" "))))


cnt = 0
with open("dataset/FB15k-237/entity2text_origin.txt", "r") as file:
    t = 0
    for line in file.readlines():
        line = line.strip().replace("\n", "")
        ids = line.split("\t")[0]
        entity_name = line.split("\t")[1]
        t += 1
        if ids not in entity2type:
            cnt += 1
            entity2type[ids] = "( )" + entity_name
            continue
        entity2type[ids] = "( " + entity2type[ids] + " ) " +entity_name

with open("dataset/FB15k-237/entity2text.txt", "w") as file:
    for ids, entity in entity2type.items():
        file.write("\t".join([ids, entity.replace("_", " ")]) + "\n")

print(f"total {cnt} entities do not has type")



