def test_can_import_pandapower_and_pandas():
    import pandas
    import pandapower.timeseries
    print(f'Pandapower version: {pandapower.__version__}')
    print(f'Pandas version: {pandas.__version__}')


