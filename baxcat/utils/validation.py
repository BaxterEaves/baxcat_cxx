
def validate_index_list(index_list, list_type='query'):
    """
    Validates that an index_list (usually either a query_list or
    constraint_list)
    """

    if not isinstance(index_list, list):
        raise TypeError('{}_indices should be a list'.format(list_type))

    for index in index_list:
        if not isinstance(index, list):
            raise TypeError('each {} in {}_indices should be a list'.format(list_type))

        if not len(index) == 2:
            raise ValueError('each {} in {}_indices should have two entries'.format(list_type))

        if not (isinstance(index[0], int) and isinstance(index[1], int)):
            raise TypeError('{} indices should be integers'.format(list_type))
