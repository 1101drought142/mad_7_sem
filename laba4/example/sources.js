const {buildSource} = require('libs/dataset/v2');

const params = Editor.getParams();
const yField = params.measureField?.[0];
const xField = params.timelineField?.[0];
const colorField = params.colorField?.[0];


const filteredOrderYears = params.OrderYear;
const filteredPaymentType = params.PaymentType;
const filteredDeliveryType = params.DeliveryType;
const where = [];

if (filteredOrderYears?.filter(item => Boolean(item.trim())).length) {
    where.push({
        column: 'OrderYear',
        type: 'title',
        operation: 'EQ',
        values: filteredOrderYears
    });
}
if (filteredPaymentType?.filter(item => Boolean(item.trim())).length) {
    where.push({
        column: 'PaymentType',
        type: 'title',
        operation: 'EQ',
        values: filteredPaymentType
    });
}
if (filteredDeliveryType?.filter(item => Boolean(item.trim())).length) {
    where.push({
        column: 'DeliveryType',
        type: 'title',
        operation: 'EQ',
        values: filteredDeliveryType
    });
}

module.exports = {
    data: buildSource({
        id: Editor.getId('dataset'),
        columns: [xField, yField, colorField].filter(Boolean),
        where,
        order_by: [{
            direction: 'asc',
            column: 'OrderYear'
        }],
    }),
};
