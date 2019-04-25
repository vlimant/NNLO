def tag_lookup(tag):
    """
    Gets the integer corresponding to the given tag string
    """
    tags = {
            'json':15,
            'result':25,
            'mbuilder':35,
            'params':45
            }
    return tags.get(tag, 0)

