## Answer to [How to impelement post-proccesing for yolo v3 or v4 onnx models in ML.Net](https://stackoverflow.com/questions/64407833/how-to-impelement-post-proccesing-for-yolo-v3-or-v4-onnx-models-in-ml-net)

I'll take the [YOLO v3 (available in the onnx/models repo)](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3) as an example. A good explaination of the model can be found [here](https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193).

First advice would be to look at the model using [Netron](https://www.electronjs.org/apps/netron). Doing so, you will see the input and output layers. They also describe these layers in the onnx/models documentation.

[Netron screenshot][1]

(I see in Netron that this particular YOLO v3 model also does some post-processing by doing the Non-maximum supression step.)

- Input layers names: `input_1`, `image_shape`
- Ouput layers names: `yolonms_layer_1/ExpandDims_1:0`, `yolonms_layer_1/ExpandDims_3:0`, `yolonms_layer_1/concat_2:0`

As per the model documentation, the input shapes are:
> Resized image (1x3x416x416) Original image size (1x2) which is [image.size['1], image.size[0]]

We first need to define the ML.Net input and output classes as follow:
```csharp
public class YoloV3BitmapData
{
	[ColumnName("bitmap")]
	[ImageType(416, 416)]
	public Bitmap Image { get; set; }

	[ColumnName("width")]
	public float ImageWidth => Image.Width;

	[ColumnName("height")]
	public float ImageHeight => Image.Height;
}

public class YoloV3Prediction
{
	/// <summary>
	/// ((52 x 52) + (26 x 26) + 13 x 13)) x 3 = 10,647.
	/// </summary>
	public const int YoloV3BboxPredictionCount = 10_647;

	/// <summary>
	/// Boxes
	/// </summary>
	[ColumnName("yolonms_layer_1/ExpandDims_1:0")]
	public float[] Boxes { get; set; }

	/// <summary>
	/// Scores
	/// </summary>
	[ColumnName("yolonms_layer_1/ExpandDims_3:0")]
	public float[] Scores { get; set; }

	/// <summary>
	/// Concat
	/// </summary>
	[ColumnName("yolonms_layer_1/concat_2:0")]
	public int[] Concat { get; set; }
}
```

We then create the ML.Net pipeline and load the prediction engine:

```csharp
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
```

**NB**: We need to define the `shapeDictionary` parameter because they are not completly defined in the model.

As per the model documentation, the output shapes are:
> The model has 3 outputs. boxes: (1x'n_candidates'x4), the coordinates of all anchor boxes, scores: (1x80x'n_candidates'), the scores of all anchor boxes per class, indices: ('nbox'x3), selected indices from the boxes tensor. The selected index format is (batch_index, class_index, box_index).

The function below will help you process the results, I leave it to you fine-tune it.

```csharp
public IReadOnlyList<YoloV3Result> GetResults(YoloV3Prediction prediction, string[] categories)
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
```

In this version of the model, they are 80 classes (see the model's GitHub documentation for the link).

You can use the above like this:
```csharp
// load image
string imageName = "dog_cat.jpg";
using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName))))
{
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

			g.DrawRectangle(Pens.Red, x1, y1, x2-x1, y2-y1);
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
```

![example](https://github.com/BobLd/YOLOv3MLNet/blob/master/YOLOV3MLNetSO/Assets/Output/cars%20road._processed.jpg)
