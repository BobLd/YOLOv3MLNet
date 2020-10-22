using Microsoft.ML.Data;

namespace YOLOV3MLNetSO.DataStructures
{
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
        public int[] Concat;
    }
}
