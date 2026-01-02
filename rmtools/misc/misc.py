def zipsort(array_to_sort_by:list, *args, reverse:bool=False)-> tuple:
    """
    Takes a list-like array to sort by, and then as many other list-likes
    after, they all have to be of the same length. Sorts all lists based on
    the first list-like, and returns a tuple of all the sorted lists.
    """

    for listlike in args:
        if len(listlike) != len(array_to_sort_by):
            raise ValueError("rmtools.zipsort: arrays of unequal length.")

    combined = sorted(zip(array_to_sort_by, *args), reverse=reverse)
    return tuple(zip(*combined))


