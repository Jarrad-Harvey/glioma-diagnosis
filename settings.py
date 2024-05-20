# Dataset properties
channel_options = {
    'T2-FLAIR': 0,
    'T1': 1,
    'T1Gd': 2,
    'T2': 3
}
slices_per_volume = 155

# Feature extraction categories
substrings = ['shape', 'firstorder', 'glcm', 'gldm', 'glrlm', 'glszm']

# Output filenames
radiomic_output_file = "radiomic_features.csv"
conventional_output_file = "conventional_features.csv"
best_features_output_file = "best_features.csv"

# Best features (Tested on volumes 1-3.)
hardcoded_top_features = {
    'intensity_features': [
        'original_firstorder_Minimum',
        'original_firstorder_Entropy',
        'original_firstorder_MeanAbsoluteDeviation',
        'original_firstorder_Uniformity',
        'original_firstorder_RobustMeanAbsoluteDeviation',
        'original_firstorder_Maximum',
        'original_firstorder_InterquartileRange',
        'original_firstorder_90Percentile',
        'original_firstorder_Range',
        'original_firstorder_RootMeanSquared'],
    'shape_features': [
        'original_shape_Elongation',
        'original_shape_Flatness',
        'original_shape_LeastAxisLength',
        'original_shape_MajorAxisLength',
        'original_shape_Maximum2DDiameterColumn',
        'original_shape_Maximum2DDiameterRow',
        'original_shape_Maximum2DDiameterSlice',
        'original_shape_Maximum3DDiameter',
        'original_shape_MeshVolume',
        'original_shape_MinorAxisLength'],
    'texture_features': [
        'original_glcm_Imc1',
        'original_glcm_Idmn',
        'original_glcm_Idn',
        'original_gldm_DependenceEntropy',
        'original_gldm_DependenceNonUniformityNormalized',
        'original_gldm_DependenceNonUniformity',
        'original_glrlm_RunEntropy',
        'original_glrlm_ShortRunEmphasis',
        'original_glszm_SmallAreaEmphasis',
        'original_glszm_ZoneEntropy'],
}