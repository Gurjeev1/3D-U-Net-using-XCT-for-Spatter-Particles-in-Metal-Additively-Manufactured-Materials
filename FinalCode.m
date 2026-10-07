classNames =["background" "SpatterParticles"];
pixelLabelID = [0 1];
TrainingDataset = "M:\Data1\CTData\gsn1n22\MAtlabCode\TrainingDataset"

SpatterLocation = fullfile(TrainingDataset, "ImagesTr");
LabelSpatterTrain=fullfile(TrainingDataset,"labelsTr");
   %SpatterTrain= imageDatastore(SpatterLocation, FileExtensions= ".tif",
   %ReadFcn=@matRead); Doesnt work


Spatter=imageDatastore(SpatterLocation,IncludeSubfolders=true,FileExtensions= ".tif", ReadFcn=@tiffreadVolume);
Label = fullfile(LabelSpatterTrain, '*.tif'); 
pxdsTrain = pixelLabelDatastore(Label,classNames,pixelLabelID,ReadFcn=@tiffreadVolume);
%LabelCombine = BioRead('Label')
%SpatterCombine = BioRead('Spatter')
%pximdsTrain = combine(Label,pxdsTrain,'Name','combinedData')

pximdsTrain = pixelLabelImageDatastore(Spatter, pxdsTrain);

ImageSize= [64 64 64];
numChannels = 1;
numClasses = 2;

unet3dNetwork = unet3d(ImageSize, numClasses, EncoderDepth=2, ConvolutionPadding="same");

figure(Units="normalized",Position=[0 0 0.5 0.55]);
plot(unet3dNetwork)
class(unet3dNetwork)
%deepNetworkDesigner(unet3dNetwork)

miniBatchSize=1

labelSize = size(read(pxdsTrain));
disp(labelSize);

function loss = modelLoss(y,targets)
    mask = ~isnan(targets);
    targets(isnan(targets)) = 0;
    loss = crossentropy(y,targets,Mask=mask);
end

options = trainingOptions("adam", ...
    MaxEpochs=25, ...
    MiniBatchSize=miniBatchSize, ...
    InitialLearnRate=1e-3, ...
    Shuffle="every-epoch", ...
    Verbose=true, ...
    Plots="training-progress");

model = trainnet(pximdsTrain, unet3dNetwork,@modelLoss,options);

ImageTesting= fullfile(TrainingDataset, "ImageTesting");
LabelsTesting= fullfile(TrainingDataset, "LabelsTesting");

voldsTest = imageDatastore(volLocTest,FileExtensions=".tif", ...
    ReadFcn=@tiffreadVolume);

pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    FileExtensions=".tif",ReadFcn=@tiffreadVolume);

imageIdx = 1;
datasetConfMat = table;
while hasdata(voldsTest)

    % Read volume and label data
    vol = read(voldsTest);
    volLabels = read(pxdsTest);

    % Create blockedImage for volume and label data
    testVolume = blockedImage(vol);
    testLabels = blockedImage(volLabels{1});

    % Calculate block metrics
    blockConfMatOneImage = apply(testVolume, ...
        @(block,labeledBlock) ...
            calculateBlockMetrics(block,labeledBlock,trainedNet,classNames), ...
        ExtraImages=testLabels, ...
        PadPartialBlocks=true, ...
        BlockSize=blockSize, ...
        BorderSize=borderSize);

    % Read all the block results of an image and update the image number
    blockConfMatOneImageDS = blockedImageDatastore(blockConfMatOneImage);
    blockConfMat = readall(blockConfMatOneImageDS);
    blockConfMat = struct2table([blockConfMat{:}]);
    blockConfMat.ImageNumber = imageIdx.*ones(height(blockConfMat),1);
    datasetConfMat = [datasetConfMat; blockConfMat];

    imageIdx = imageIdx + 1;
end

[metrics,blockMetrics] = evaluateSemanticSegmentation( ...
    datasetConfMat,classNames,Metrics="all");