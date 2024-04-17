import { GPU } from "gpu.js";
import { real_main_with_make_propergate_fn } from "../../pkg/waveform_function_collapse_lib";

/*
const gpu = new GPU({
    mode: "webgl"
});
*/

export function makePropergateFn(sourceMapRows: number, sourceMapCols: number, targetMapRows: number, targetMapColumns: number, numUniqueTiles: number): (sourceMap: number[][], targetMap: number[][][]) => number[][][] {
    let impl = function(sourceMap: number[][], targetMap: number[][][]) {
        let targetRow = this.thread.z;
        let targetCol = this.thread.y;
        let targetTileIndex = this.thread.x;
        let targetVal = targetMap[targetRow][targetCol][targetTileIndex];
        let result = 0.0;
        if (targetVal != 0.0) {
            for (let sourceRow = 1; sourceRow < (this.constants.sourceMapRows as number)-1; sourceRow++) {
                for (let sourceCol = 1; sourceCol < (this.constants.sourceMapCols as number)-1; sourceCol++) {
                    let sourceVal = sourceMap[sourceRow][sourceCol];
                    if (sourceVal == targetTileIndex) {
                        let match = true;
                        for (let i = -1; i <= 1; i++) {
                            for (let j = -1; j <= 1; j++) {
                                if (i == 0 && j == 0) {
                                    continue;
                                }
                                let k = targetRow + i;
                                let l = targetCol + j;
                                let m = sourceRow + i;
                                let n = sourceRow + j;
                                if (k >= 0 && k < (this.constants.targetMapRows as number)) {
                                    if (l >= 0 && l < (this.constants.targetMapColumns as number)) {
                                        let tile = sourceMap[m][n];
                                        if (targetMap[k][l][tile] == 0) {
                                            match = false;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!match) {
                                break;
                            }
                        }
                        if (match) {
                            result += 1.0;
                        }
                    }
                }
            }
        }
        return result;
    };
    /*
    const propergate = gpu.createKernel(impl, {
        output: [numUniqueTiles, targetMapColumns, targetMapRows],
        constants: {
            sourceMapRows,
            sourceMapCols,
            targetMapRows,
            targetMapColumns,
            numUniqueTiles,
        },
    });
    return (sourceMap: number[][], targetMap: number[][][]) => propergate(sourceMap, targetMap) as number[][][];
    */
    return (sourceMap: number[][], targetMap: number[][][]) => {
        let result: number[][][] = [];
        let context = {
            thread: {
                x: 0,
                y: 0,
                z: 0,
            },
            constants: {
                sourceMapRows,
                sourceMapCols,
                targetMapRows,
                targetMapColumns,
                numUniqueTiles,
            },
        };
        for (let i = 0; i < targetMapRows; i++) {
            let row: number[][] = [];
            context.thread.z = i;
            for (let j = 0; j < targetMapColumns; j++) {
                let col: number[] = [];
                context.thread.y = j;
                for (let k = 0; k < numUniqueTiles; k++) {
                    context.thread.x = k;
                    col.push(impl.call(context, sourceMap, targetMap));
                }
                row.push(col);
            }
            result.push(row);
        }
        return result;
    };
}

real_main_with_make_propergate_fn(makePropergateFn);
