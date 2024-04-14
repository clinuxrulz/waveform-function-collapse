import { GPU } from "gpu.js";

const gpu = new GPU();

export function makePropergateFn(sourceMapRows: number, sourceMapCols: number, targetMapRows: number, targetMapColumns: number, numUniqueTiles: number): (sourceMap: number[][], targetMap: number[][][]) => number[][][] {
    const propergate = gpu.createKernel(function(sourceMap: number[][], targetMap: number[][][]) {
        let targetRow = this.thread.y;
        let targetCol = this.thread.x;
        let targetTileIndex = this.thread.z;
        let targetVal = targetMap[targetRow][targetCol][targetTileIndex];
        if (targetVal == 0.0) {
            return 0.0;
        }
        let result = 0.0;
        for (let sourceRow = 0; sourceRow < (this.constants.sourceMapRows as number); sourceRow++) {
            for (let sourceCol = 0; sourceCol < (this.constants.sourceMapCols as number); sourceCol++) {
                let sourceVal = sourceMap[sourceRow][sourceCol];
                if (sourceVal == targetTileIndex) {
                    if (sourceRow > 0) {
                        let sourceUp = sourceMap[sourceRow - 1][sourceCol];
                        if (targetRow > 0) {
                            if (targetMap[targetRow - 1][targetCol][sourceUp] != 0.0) {
                                result += 1.0;
                            }
                        } else {
                            result += 1.0;
                        }
                    }
                    if (sourceRow < (this.constants.sourceMapRows as number) - 1) {
                        let sourceDown = sourceMap[sourceRow + 1][sourceCol];
                        if (targetRow < (this.constants.targetMapRows as number) - 1) {
                            if (targetMap[targetRow + 1][targetCol][sourceDown] != 0.0) {
                                result += 1.0;
                            }
                        } else {
                            result += 1.0;
                        }
                    }
                    if (sourceCol > 0) {
                        let sourceLeft = sourceMap[sourceRow][sourceCol - 1];
                        if (targetCol > 0) {
                            if (targetMap[targetRow][targetCol - 1][sourceLeft] != 0.0) {
                                result += 1.0;
                            }
                        } else {
                            result += 1.0;
                        }
                    }
                    if (sourceCol < (this.constants.sourceMapCols as number) - 1) {
                        let sourceRight = sourceMap[sourceRow][sourceCol + 1];
                        if (targetCol < (this.constants.targetMapColumns as number) - 1) {
                            if (targetMap[targetRow][targetCol + 1][sourceRight] != 0.0) {
                                result += 1.0;
                            }
                        } else {
                            result += 1.0;
                        }
                    }
                }
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
    return (sourceMap: number[][], targetMap: number[][][]) => propergate(sourceMap, targetMap) as number[][][];
}
