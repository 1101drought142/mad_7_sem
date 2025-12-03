const Dataset = require('libs/dataset/v2');
const loadedData = Editor.getLoadedData();
const data = Dataset.processData(loadedData, 'data', Editor);

const params = Editor.getParams();

const yField = params.measureField?.[0];
const xField = params.timelineField?.[0];
const colorField = params.colorField?.[0];
const stacking = params.stacking?.[0];

function groupBy(arr, field) {
    const grouped = arr.reduce((acc, item) => {
        const key = item[field];
        if (!acc[key]) {
            acc[key] = [];
        }
        acc[key].push(item);
        
        return acc;
    }, {});

    return Object.values(grouped);
}

const categories = Array.from(new Set(data.map(d => String(d[xField]))));

module.exports = {
    chart: {
        margin: {
            left: 12,
            right: 12,
            top: 18,
            bottom: 18,
        }
    },
    series: {
        data: (colorField ? groupBy(data, colorField) : [data]).map((arr) => {
            const seriesData = arr
                .map((d, index) => {
                    let actionParamsConfig = {
                        [xField]: d[xField],
                    };
                    const hasExtraParams = Object.keys(d).length > 2;
                    if (hasExtraParams) {
                        const extraParams = {};
                        Object.entries(d).forEach(([key, value]) => {
                            if (key !== xField && key !== yField) {
                                extraParams[key] = value;
                            }
                        });

                        actionParamsConfig = {...actionParamsConfig, ...extraParams};
                    }

                    return {
                        x: index, 
                        y: d[yField],
                        custom: {
                            actionParams: actionParamsConfig,
                        }
                    };
                });
            const name = colorField ? arr[0][colorField] : yField;

            return {
                type: 'bar-x',
                name,
                data: seriesData,
                stacking,
            };
        }),
    },
    xAxis: {
        type: 'category',
        categories,
    },
    yAxis: [{
        ticks: {
            pixelInterval: 100,
        },
        title: {
            text: yField
        }
    }]
};
