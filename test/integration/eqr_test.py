from pudl_rmi.process_eqr import main, ArgsPatch



def test_main():
    main(ArgsPatch())
    assert False