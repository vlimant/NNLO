def tag_lookup(tag):
    """
    Gets the integer corresponding to the given tag string
    """
    tags = {
            'json':1,
            'result':2,
            'mbuilder':3,
            'params':4
            }
    return tags.get(tag, 0)

