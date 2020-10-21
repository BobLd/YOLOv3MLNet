using Microsoft.ML;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using YOLOv3MLNet.DataStructures;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace YOLOv3MLNet
{
    class Program
    {
        const string modelLocation = @"D:\MachineLearning\Document Layout Analysis\notebooks\yolo\dla_yolov3.onnx";

        const string imageFolder = @"Assets\Images";

        const string imageOutputFolder = @"Assets\Output";

        static readonly string[] classesNames = new string[]
        {
            "caption", "chart", "credit",
            "drop-capital", "floating", "footer",
            "frame", "graphics", "header",
            "heading", "image", "linedrawing",
            "maths", "noise", "page-number",
            "paragraph", "separator", "table"
        };

        static float confidenceThreshold = 0.2f;

        static float confidenceIou = 0.5f;

        static void Main()
        {
            Directory.CreateDirectory(imageOutputFolder);
            MLContext mlContext = new MLContext();

            // load prediction engine
            var predictionEngine = LoadPredictionEngine(mlContext, modelLocation);

            // load image
            string imageName = "PMC5055614_00001.jpg";
            using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName))))
            {
                // predict
                var results = predictionEngine.Predict(new YoloV3BitmapData() { Image = bitmap })
                                              .GetResults(classesNames, confidenceThreshold, confidenceIou);

                // draw predictions
                using (var g = Graphics.FromImage(bitmap))
                {
                    foreach (var result in results)
                    {
                        var x1 = result.BBox[0];
                        var y1 = result.BBox[1];
                        var w = result.BBox[2] - x1;
                        var h = result.BBox[3] - y1;

                        g.DrawRectangle(Pens.Red, x1, y1, w, h);
                        using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                        {
                            g.FillRectangle(brushes, x1, y1, w, h);
                        }

                        g.DrawString(result.Label + " " + result.Confidence.ToString("0.00"),
                                     new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                    }

                    bitmap.Save(Path.Combine(imageOutputFolder, Path.ChangeExtension(imageName, "_processed" + Path.GetExtension(imageName))));
                }
            }
        }

        public static PredictionEngine<YoloV3BitmapData, YoloV3Prediction> LoadPredictionEngine(MLContext mlContext, string modelPath)
        {
            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "image", imageWidth: 416, imageHeight: 416, resizing: ResizingKind.IsoPad)
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image", scaleImage: 1f / 255f))
                            .Append(mlContext.Transforms.ApplyOnnxModel(inputColumnNames: new[] { "image" }, outputColumnNames: new[] { "bboxes", "classes" }, modelFile: modelPath));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV3BitmapData>()));

            // Create prediction engine
            return mlContext.Model.CreatePredictionEngine<YoloV3BitmapData, YoloV3Prediction>(model);
        }
    }
}
