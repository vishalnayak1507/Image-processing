for i = 4:5
    % File paths
    inputImage = sprintf('data/im%d.png', i);
    compressedImage = sprintf('data/m_im%d.pbm', i);
    residualImage = sprintf('data/res_im%d.png', i);
    restoredImage = sprintf('data/restored_im%d.png', i);

    % Encoder step
    encoder(inputImage, compressedImage, residualImage);

    % Decoder step
    % decoder(compressedImage, residualImage, inputImage, restoredImage);
end
