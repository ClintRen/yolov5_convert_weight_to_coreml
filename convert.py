import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolov5-repo",
        required=True,
        # default="",
        help="path to yolov5 repo",
    )
    parser.add_argument(
        "--weight", type=str, default="yolov5s.pt", help="yolov5 weight path"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="image input size (pixels)",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--quantize",
        default=False,
        action="store_true",
        help="quantize model to FP16 and Int8",
    )

    # Other model options
    parser.add_argument("--description", default="yolov5s", help="model description")
    parser.add_argument("--author", default="yolov5", help="model author")
    parser.add_argument("--version", default="6.2", help="model version")
    parser.add_argument("--license", default="GPL-3.0", help="model license")
    args = parser.parse_args()
    sys.path.insert(0, args.yolov5_repo)

    import types
    import models
    from utils.general import check_img_size
    from utils.activations import Hardswish, SiLU
    from utils.torch_utils import select_device
    from models.experimental import attempt_load
    import coremltools.proto.FeatureTypes_pb2 as ft
    from conversion_modules import *

    weight = args.weight
    img_size = [args.img_size] * 2
    iouThreshold = args.iou_thres
    confidenceThreshold = args.conf_thres
    quantize = args.quantize

    device = select_device(args.device)  # cpu or cuda

    model = attempt_load(weight)  # load FP32 model
    labels = model.names
    # check model
    gs = int(max(model.stride))  # grid size (max stride)
    img_size = [
        check_img_size(x, gs) for x in img_size
    ]  # verify img_size are gs-multiples

    # input
    im = torch.zeros(1, 3, *img_size).to(device)  # init img

    # update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.7.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, models.yolo.Detect):
            m.inplace = False
            m.forward = types.MethodType(detect_export_forward, m)

    for _ in range(2):
        y = model(im)  # dry run

    # export model
    export_model = ExportModel(model, img_size=img_size)

    num_boxes = y[0].shape[1]
    num_classes = y[0].shape[2] - 5

    # convert to torchscript
    ts = torch.jit.trace(export_model, im, strict=False)

    # CoreML model export
    # convert model from torchscript and apply pixel scaling as per detect.py
    coreml_model = ct.convert(
        ts,
        inputs=[
            ct.ImageType(name="input", shape=im.shape, scale=1 / 255.0, bias=[0, 0, 0])
        ],
    )
    spec = coreml_model.get_spec()

    old_scores_output_name = spec.description.output[0].name
    old_box_output_name = spec.description.output[1].name
    ct.utils.rename_feature(spec, old_scores_output_name, "raw_confidence")
    ct.utils.rename_feature(spec, old_box_output_name, "raw_coordinates")
    spec.description.output[0].type.multiArrayType.shape.extend(
        [num_boxes, num_classes]
    )
    spec.description.output[1].type.multiArrayType.shape.extend([num_boxes, 4])
    spec.description.output[0].type.multiArrayType.dataType = ft.ArrayFeatureType.DOUBLE
    spec.description.output[1].type.multiArrayType.dataType = ft.ArrayFeatureType.DOUBLE

    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
    spec.description
    nmsSpec = createNmsModelSpec(
        builder.spec, num_classes, labels, iouThreshold, confidenceThreshold
    )
    # run the functions to add decode layer and NMS to the model.
    nms_model = ct.models.MLModel(nmsSpec)
    combineModelsAndExport(
        builder.spec,
        nmsSpec,
        weight.replace(".pt", ".mlmodel"),
        img_size,
        iouThreshold,
        confidenceThreshold,
        quantize,
        args.description,
        args.author,
        args.version,
        args.license,
    )  # The model will be saved in this path.
