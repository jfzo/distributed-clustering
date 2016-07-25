function h = tabularHeatMap(data, varargin)
%% tabularHeatMap
% Author: Thomas Churchman
% Date: 2015/10/14
% 
% Plot a matrix as a tabular heat map. 
% 
% INPUTS
%   data (required) - The matrix to plot as a tabular heat map
%   OPTIONS (Optional) - Optional arguments as key-value pairs
%     Key            | Value
%     ----------------------
%     'TextLabels'   | Boolean indicating whether textual labels for the
%                    | values should be printed (default: true)
%     'Colorbar'     | Boolean indicating whether a colorbar should be
%                    | shown
%     'Colormap'     | The colormap to use for the heat map color values,
%                    | either a reference to a built-in colormap or a 
%                    | m x 3 colormap matrix (default: 'jet')
%
% OUTPUTS
%   h - The figure handle
% 
% EXAMPLE 1
% A = magic(10);
% tabularHeatMap(A);
% 
% EXAMPLE 2
% confusion = crosstab(responses, correctAnswers);
% h = tabularHeatMap(confusion, 'Colormap', 'winter');
% title('Confusion matrix');
% xlabel('Correct');
% ylabel('Response');
% h.XAxisLocation = 'top';
% h.XTick = [1 2 3];
% h.XTickLabel = {'A', 'B', 'C'};
% h.YTick = [1 2 3];
% h.YTickLabel = {'A', 'B', 'C'};

    % Parse optional arguments
    p = inputParser;
    
    defaultTextLabels = true;
    defaultColorbar = true;
    defaultColormap = 'jet';
    
    addOptional(p, 'TextLabels', defaultTextLabels, @islogical);
    addOptional(p, 'Colorbar', defaultColorbar, @islogical);
    addOptional(p, 'Colormap', defaultColormap);
    
    parse(p,varargin{:});
    
    % Set colormap
    colormap(p.Results.Colormap);
    
    % Show data as image (i.e., heat map)
    imagesc(data);
    
    if p.Results.Colorbar
        % Show colorbar
        colorbar;
    end
    
    h = gca;
    
    if p.Results.TextLabels
        % Show textual labels (i.e., making it a tabular heat map)
        textStrings = num2str(data(:));
        textStrings = strtrim(cellstr(textStrings));
        [x, y] = meshgrid(1:size(data,2), 1:size(data,1));
        hText = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
        
        % Find colors of data points
        cm = colormap;
        cLim = get(h,'CLim');
        % Find colormap indices of data point values (i.e. normalize to 
        % [1, n] with n the length of the colormap).
        colorsIdx = fix((data-cLim(1))/(cLim(2)-cLim(1)) .* (size(cm,1)-1)) + 1;
        % Get the RGB colors the indices represent
        colors = arrayfun(@(idx) cm(idx,:), colorsIdx, 'UniformOutput', false);
        colors = colors(:);

        % Set text color depending on whether the data point is colored
        % light or dark for better contrast
        for i=1:numel(data)
            d = mean(colors{i});
            
            if d < 0.4
                % Data point is dark
                hText(i).Color = 'white';
            else
                % Data point is light
                hText(i).Color = 'black';
            end
        end
    end
end

