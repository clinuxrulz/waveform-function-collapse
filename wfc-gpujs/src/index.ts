import { GPU } from "gpu.js";

const gpu = new GPU();

export function makePropergateFn(sourceMapRows: number, sourceMapCols: number, targetMapRows: number, targetMapColumns: number, numUniqueTiles: number): (sourceMap: number[][], targetMap: number[][]) => number[][] {
    const propergate = gpu.createKernel(function(sourceMap: number[][], targetMap: number[][]) {
        let targetRow = this.thread.y;
        let targetCol = this.thread.x;
        let targetVal = targetMap[targetRow][targetCol];
        let targetUp: number;
        if (targetRow > 0) {
            targetUp = targetMap[targetRow - 1][targetCol];
        } else {
            targetUp = 0;
        }
        let targetDown: number;
        if (targetRow < (this.constants.targetMapRows as number) - 1) {
            targetDown = targetMap[targetRow + 1][targetCol];
        } else {
            targetDown = 0;
        }
        let targetLeft: number;
        if (targetCol > 0) {
            targetLeft = targetMap[targetRow][targetCol - 1];
        } else {
            targetLeft = 0;
        }
        let targetRight: number;
        if (targetCol < (this.constants.targetMapColumns as number) - 1) {
            targetRight = targetMap[targetRow][targetCol + 1];
        } else {
            targetRight = 0;
        }
        let result = targetVal;
        for (let tileIndex = numUniqueTiles-1; tileIndex >= 0; tileIndex--) {
            
        }
        for (let sourceRow = 0; sourceRow < (this.constants.sourceMapRows as number); sourceRow++) {
            for (let sourceCol = 0; sourceCol < (this.constants.sourceMapCols as number); sourceCol++) {

            }
        }
        return result;
    }, {
        output: [targetMapRows, targetMapColumns],
        constants: {
            sourceMapRows,
            sourceMapCols,
            targetMapRows,
            targetMapColumns,
            numUniqueTiles,
        },
    });
    return (sourceMap: number[][], targetMap: number[][]) => propergate(sourceMap, targetMap) as number[][];
}
