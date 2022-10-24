import torch
import torch.nn as nn
import coremltools as ct


class ExportModel(nn.Module):
    def __init__(self, base_model, img_size):
        super(ExportModel, self).__init__()
        self.base_model = base_model
        self.img_size = img_size

    def forward(self, x):
        x = self.base_model(x)[0]
        x = x.squeeze(0)
        # Convert box coords to normalized coordinates [0 ... 1]
        w = self.img_size[0]
        h = self.img_size[1]
        objectness = x[:, 4:5]
        class_probs = x[:, 5:] * objectness
        boxes = x[:, :4] * torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])
        return class_probs, boxes


def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((ny, nx, 2)).float()


def detect_export_forward(self, x):
    z = []  # inference output
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
        bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[i] = (
            x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        )

        if not self.training:  # inference
            if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            y = x[i].sigmoid()
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                xy, wh, conf = y.split(
                    (2, 2, self.nc + 1), 4
                )  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, -1, self.no))

    return torch.cat(z, 1), x


def createNmsModelSpec(
    nnSpec, numberOfClassLabels, classLabels, iouThreshold, confidenceThreshold
):
    """
    Create a coreml model with nms to filter the results of the model
    """
    nmsSpec = ct.proto.Model_pb2.Model()
    nmsSpec.specificationVersion = 4

    # Define input and outputs of the model
    for i in range(2):
        nnOutput = nnSpec.description.output[i].SerializeToString()

        nmsSpec.description.input.add()
        nmsSpec.description.input[i].ParseFromString(nnOutput)

        nmsSpec.description.output.add()
        nmsSpec.description.output[i].ParseFromString(nnOutput)

    nmsSpec.description.output[0].name = "confidence"
    nmsSpec.description.output[1].name = "coordinates"

    # Define output shape of the model
    outputSizes = [numberOfClassLabels, 4]
    for i in range(len(outputSizes)):
        maType = nmsSpec.description.output[i].type.multiArrayType
        # First dimension of both output is the number of boxes, which should be flexible
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[0].lowerBound = 0
        maType.shapeRange.sizeRanges[0].upperBound = -1
        # Second dimension is fixed, for "confidence" it's the number of classes, for coordinates it's position (x, y) and size (w, h)
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[1].lowerBound = outputSizes[i]
        maType.shapeRange.sizeRanges[1].upperBound = outputSizes[i]
        del maType.shape[:]

    # Define the model type non maximum supression
    nms = nmsSpec.nonMaximumSuppression
    nms.confidenceInputFeatureName = "raw_confidence"
    nms.coordinatesInputFeatureName = "raw_coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    # Some good default values for the two additional inputs, can be overwritten when using the model
    nms.iouThreshold = iouThreshold
    nms.confidenceThreshold = confidenceThreshold
    nms.stringClassLabels.vector.extend(classLabels)

    return nmsSpec


# Just run to combine the model added decode and the NMS.
def combineModelsAndExport(
    builderSpec,
    nmsSpec,
    fileName,
    img_size,
    iouThreshold,
    confidenceThreshold,
    quantize,
    description,
    author,
    version,
    license,
):
    """
    Combines the coreml model with export logic and the nms to one final model. Optionally save with different quantization (32, 16, 8)
    """
    try:
        print(f"Combine CoreMl model with nms and export model")
        # Combine models to a single one
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, img_size[0], img_size[1])),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=["confidence", "coordinates"],
        )

        # Required version (>= ios13) in order for mns to work
        pipeline.spec.specificationVersion = 4

        pipeline.add_model(builderSpec)
        pipeline.add_model(nmsSpec)

        pipeline.spec.description.input[0].ParseFromString(
            builderSpec.description.input[0].SerializeToString()
        )
        pipeline.spec.description.output[0].ParseFromString(
            nmsSpec.description.output[0].SerializeToString()
        )
        pipeline.spec.description.output[1].ParseFromString(
            nmsSpec.description.output[1].SerializeToString()
        )
        pipeline.spec.description.input[
            0
        ].shortDescription = "Image to detect objects in"

        # Metadata for the model‚
        pipeline.spec.description.input[
            1
        ].shortDescription = (
            f"(optional) IOU Threshold override (Default: {iouThreshold})"
        )
        pipeline.spec.description.input[
            2
        ].shortDescription = (
            f"(optional) Confidence Threshold override (Default: {confidenceThreshold})"
        )
        pipeline.spec.description.output[
            0
        ].shortDescription = "Boxes \xd7 Class confidence"
        pipeline.spec.description.output[
            1
        ].shortDescription = "Boxes \xd7 [x, y, width, height] (Normalized)"
        pipeline.spec.description.metadata.shortDescription = description
        pipeline.spec.description.metadata.author = author
        pipeline.spec.description.metadata.versionString = version
        pipeline.spec.description.metadata.license = license
        # Add the list of class labels and the default threshold values too.
        # user_defined_metadata = {
        #     "iou_threshold": str(iouThreshold),
        #     "confidence_threshold": str(confidenceThreshold),
        # }
        # pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)

        model = ct.models.MLModel(pipeline.spec)
        model.save(fileName)
        print(f"CoreML export success, saved as {fileName}")

        if quantize:
            fileName16 = fileName.replace(".mlmodel", "_FP16.mlmodel")
            modelFp16 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=16
            )
            modelFp16.save(fileName16)
            print(f"CoreML export success, saved as {fileName16}")

            fileName8 = fileName.replace(".mlmodel", "_Int8.mlmodel")
            modelFp8 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8
            )
            modelFp8.save(fileName8)
            print(f"CoreML export success, saved as {fileName8}")
    except Exception as e:
        print(f"CoreML export failure: {e}")
