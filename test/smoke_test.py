def test_can_import_pandapower_and_pandas():
    import pandapower
    import pandas
    print(f'Pandapower version: {pandapower.__version__}')
    print(f'Pandas version: {pandas.__version__}')
