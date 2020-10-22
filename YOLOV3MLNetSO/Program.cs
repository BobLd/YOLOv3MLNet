using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using YOLOV3MLNetSO.DataStructures;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace YOLOV3MLNetSO
{
    /*
     * My answer to https://stackoverflow.com/questions/64407833/how-to-impelement-post-proccesing-for-yolo-v3-or-v4-onnx-models-in-ml-net
     */
    class Program
    {
        const string imageFolder = @"Assets\Images";

        const string imageOutputFolder = @"Assets\Output";

        static readonly string[] classesNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

        static void Main(string[] args)
        {
            Directory.CreateDirectory(imageOutputFolder);
            MLContext mlContext = new MLContext();

            // model is available here:
            // https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "input_1", imageWidth: 416, imageHeight: 416, resizing: ResizingKind.IsoPad)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1", outputAsFloatArray: true, scaleImage: 1f / 255f))
                .Append(mlContext.Transforms.Concatenate("image_shape", "height", "width"))
                .Append(mlContext.Transforms.ApplyOnnxModel(shapeDictionary: new Dictionary<string, int[]>() { { "input_1", new[] { 1, 3, 416, 416 } } },
                                inputColumnNames: new[]
                                {
                                    "input_1",
                                    "image_shape"
                                },
                                outputColumnNames: new[]
                                {
                                    "yolonms_layer_1/ExpandDims_1:0",
                                    "yolonms_layer_1/ExpandDims_3:0",
                                    "yolonms_layer_1/concat_2:0"
                                },
                                modelFile: @"D:\yolov3-10.onnx"));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV3BitmapData>()));

            // Create prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV3BitmapData, YoloV3Prediction>(model);

            // load image
            string imageName = "cars road.jpg";
            using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName))))
            {
                var preview = model.Preview(mlContext.Data.LoadFromEnumerable(new List<YoloV3BitmapData>() { new YoloV3BitmapData() { Image = bitmap } }));
                // predict
                var predict = predictionEngine.Predict(new YoloV3BitmapData() { Image = bitmap });
                var results = GetResults(predict, classesNames);

                // draw predictions
                using (var g = Graphics.FromImage(bitmap))
                {
                    foreach (var result in results)
                    {
                        var y1 = result.BBox[0];
                        var x1 = result.BBox[1];
                        var y2 = result.BBox[2];
                        var x2 = result.BBox[3];

                        g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                        using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                        {
                            g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                        }

                        g.DrawString(result.Label + " " + result.Confidence.ToString("0.00"),
                                     new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                    }

                    bitmap.Save(Path.Combine(imageOutputFolder, Path.ChangeExtension(imageName, "_processed" + Path.GetExtension(imageName))));
                }
            }
        }

        public static IReadOnlyList<YoloV3Result> GetResults(YoloV3Prediction prediction, string[] categories)
        {
            if (prediction.Concat == null || prediction.Concat.Length == 0)
            {
                return new List<YoloV3Result>();
            }

            if (prediction.Boxes.Length != YoloV3Prediction.YoloV3BboxPredictionCount * 4)
            {
                throw new ArgumentException();
            }

            if (prediction.Scores.Length != YoloV3Prediction.YoloV3BboxPredictionCount * categories.Length)
            {
                throw new ArgumentException();
            }

            List<YoloV3Result> results = new List<YoloV3Result>();

            // Concat size is 'nbox'x3 (batch_index, class_index, box_index)
            int resulstCount = prediction.Concat.Length / 3;
            for (int c = 0; c < resulstCount; c++)
            {
                var res = prediction.Concat.Skip(c * 3).Take(3).ToArray();

                var batch_index = res[0];
                var class_index = res[1];
                var box_index = res[2];

                var label = categories[class_index];
                var bbox = new float[]
                {
                    prediction.Boxes[box_index * 4],
                    prediction.Boxes[box_index * 4 + 1],
                    prediction.Boxes[box_index * 4 + 2],
                    prediction.Boxes[box_index * 4 + 3],
                };
                var score = prediction.Scores[box_index + class_index * YoloV3Prediction.YoloV3BboxPredictionCount];

                results.Add(new YoloV3Result(bbox, label, score));
            }

            return results;
        }

    }
}
