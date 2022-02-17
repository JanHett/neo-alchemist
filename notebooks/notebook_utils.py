from matplotlib.pyplot import imshow
import PyOpenColorIO as OCIO

ocio_config = OCIO.GetCurrentConfig()

ocio_display = ocio_config.getDefaultDisplay()
ocio_view = ocio_config.getDefaultView(ocio_display)
ocio_processor = ocio_config.getProcessor(
    OCIO.ROLE_SCENE_LINEAR,
    ocio_display,
    ocio_view,
    OCIO.TRANSFORM_DIR_FORWARD)
ocio_cpu = ocio_processor.getDefaultCPUProcessor()

def ccshow(image):
    """
    Applies a scene-linear to display colour space transform to a copy of the
    image before calling `imshow` on it
    """
    to_display = image.copy()
    ocio_cpu.applyRGB(to_display)

    imshow(to_display)