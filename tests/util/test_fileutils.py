from minerva_lib.util.fileutils import FileUtils

def test_pyramid_levels():
    files = [
        "C0-T0-Z0-L0-Y0-X0.png",
        "C1-T0-Z0-L1-Y0-X0.png",
        "C2-T1-Z0-L2-Y1-X2.png",
        "C0-T0-Z0-L1-Y0-X0.png"
    ]
    pyramid_levels = FileUtils.get_pyramid_levels(files)
    assert(pyramid_levels == 3)