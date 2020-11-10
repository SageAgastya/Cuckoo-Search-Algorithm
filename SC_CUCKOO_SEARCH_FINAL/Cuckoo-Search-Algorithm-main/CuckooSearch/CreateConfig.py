import json

dictionary = {
    "P_a":0.25,
    "beta":1.5,
    "alpha":1,
    "lb":-5,
    "ub":5,
    "dim":2,
    "nests":5,
    "max_generations":10
}

# Serializing json
json_object = json.dumps(dictionary)

# Writing to config.json
with open("config.json", "w") as outfile:
    outfile.write(json_object)