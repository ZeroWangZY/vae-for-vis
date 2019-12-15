let scagnosticIndices = (function () {

  function getDomain(dataset) {
    let minX = Infinity,  maxX = -Infinity, 
        minY = Infinity,  maxY = -Infinity;

    for(data of dataset) {
      let {x, y} = data;
      minX = x < minX ? x : minX;
      maxX = x > maxX ? x : maxX;
      minY = y < minY ? y : minY;
      maxY = y > maxY ? y : maxY;
    }

    return {
      xDomain: [minX, maxX],
      yDomain: [minY, maxY]
    }
  }

  function _splitDataset(dataset) {
    let xDataset = [], 
        yDataset = [];

    for(data of dataset) {
      xDataset.push(data.x);
      yDataset.push(data.y);
    }
    
    return {
      xDataset: xDataset,
      yDataset: yDataset
    };
  }

  function _mergeDataset(xDataset, yDataset) {
    if (xDataset.length !== yDataset.length) {
      console.error("dataset X and dataset Y must have the same length!");
    }

    let dataset = [];
    for(let i = 0; i < xDataset.length; i += 1) {
      dataset.push({
        x: xDataset[i],
        y: yDataset[i]
      });
    }
    return dataset;
  }

  function _normalize(translation, factor, _dataset) {
    for(let i = 0; i < _dataset.length; i += 1) {
      _dataset[i] += translation;
      if (factor !== 0) {
        _dataset[i] /= factor;
      }
    }
    return _dataset;
  }

  function deduplicateDataset(dataset) {
    let set = new Set();
    let newDataset = [];

    dataset.forEach(data => {
      set.add(`${data.x}-${data.y}`);
    })

    set.forEach(dataStr => {
      let [x, y] = dataStr.split("-").map(d => Number(d));
      newDataset.push({
        x,
        y
      });
    });

    return newDataset;
  }

  function normalizeDataset(dataset, type = "Separate") {
    let xTranslation, xFactor,
        yTranslation, yFactor,
        length = dataset.length,
        {xDomain, yDomain} = getDomain(dataset),
        xSpan = xDomain[1] - xDomain[0],
        ySpan = yDomain[1] - yDomain[0],
        {xDataset, yDataset} = _splitDataset(dataset);

    if (type === "Separate") {
      xTranslation = -xDomain[0];
      xFactor = xSpan;
      yTranslation = -yDomain[0];
      yFactor = ySpan;
    } else {
      xTranslation = -xDomain[0];
      xFactor = max(xSpan, ySpan);
      yTranslation = -yDomain[0];
      yFactor = xFactor;
    }
    xDataset = _normalize(xTranslation, xFactor, xDataset);
    yDataset = _normalize(yTranslation, yFactor, yDataset);

    return _mergeDataset(xDataset, yDataset);
  }

  function _getEdgeLength(point1, point2, type = "Euclidean") {
    let edgeLength;
    switch(type) {
      case "Euclidean":
        edgeLength = Math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2);
        break;
    }
    return edgeLength;
  }

  function _getGraphLength(mst) {
    let graphLength = 0;
    for(let i = 1; i < mst.adjVex.length; i += 1) {
      graphLength += mst.adjLength[i];
    }
    return graphLength;
  }

  function _getAdjMatrix(dataset) {
    let edgeLength, adjMatrix = [], innerList = [];

    for(let i = 0; i < dataset.length; i += 1) {
      innerList.push(Infinity);
    }
    for(let i = 0; i < dataset.length; i += 1) {
      adjMatrix.push([...innerList]);
      adjMatrix[i][i] = 0;
    }

    for(let i = 0; i < dataset.length; i += 1) {
      for(let j = i + 1; j < dataset.length; j += 1) {
        edgeLength = _getEdgeLength(dataset[i], dataset[j]);
        adjMatrix[i][j] = edgeLength;
        adjMatrix[j][i] = edgeLength;
      }
    }
    return adjMatrix;
  }

  function _getMSTPrim(adjMatrix) {
    let min, i, j, k,
        adjVex = [], lowCost = [], adjLength = [];

    for(i = 0; i < adjMatrix.length; i += 1) {
      adjVex.push(0);
      lowCost.push(adjMatrix[0][i]);
      adjLength.push(0);
    }

    for(i = 1; i < adjMatrix.length; i += 1) {
      min = Infinity;
      j = 1; k = 0;
      while(j < adjMatrix.length) {
        if (lowCost[j] != 0 && lowCost[j] < min) {
          min = lowCost[j];
          k = j;
        }
        j += 1;
      }
      adjLength[k] = lowCost[k];
      lowCost[k] = 0;
      for(j = 1; j < adjMatrix.length; j += 1) {
        if (lowCost[j] != 0 && adjMatrix[k][j] < lowCost[j]) {
          lowCost[j] = adjMatrix[k][j];
          adjVex[j] = k;
        }
      }
    }

    return {
      adjVex: adjVex,
      adjLength: adjLength
    };   
  }

  function _getOmega(mst) {
    let [tmp, ...tmpAdjLength] = mst.adjLength;

    tmpAdjLength.sort((a, b) => a - b);

    let q75 = tmpAdjLength[Math.round(tmpAdjLength.length * 0.75) - 1],
        q25 = tmpAdjLength[Math.round(tmpAdjLength.length * 0.25) - 1 > 0 ?
          Math.round(tmpAdjLength.length * 0.25) - 1 : 0];

    return q75 + 1.5 * (q75 - q25);
  }

  function _getAHArea(dataset) {
    let mst = _getMSTPrim(_getAdjMatrix(dataset)),
        alpha = _getOmega(mst),
        alphaHull = _getAlphaHull(dataset, alpha);
    return _getHullArea(alphaHull);
  }

  function _getCHArea(dataset) {
    let convexHull = _getConvexHull(dataset);
    return _getHullArea(convexHull);
  }

  function _getHullArea(hull) {
    let area = 0, j = hull.length - 1;

    for(let i = 0; i < hull.length; i += 1) {
      area += (hull[j].x + hull[i].x) * (hull[j].y - hull[i].y);
      j = i;
    }

    return 0.5 * Math.abs(area);
  }

  function _formatDataset(dataset) {
    let newDataset = [];
    for(let i = 0; i < dataset.length; i += 1) {
      newDataset.push([dataset[i].x, dataset[i].y]);
    }
    return newDataset;
  }

  function _getAlphaHull(dataset, alpha) {
    let newDataset = _formatDataset(dataset),
        hull = _getAlphaHullLib(newDataset, alpha),
        alphaHull = [];

    for(let i = 0; i < hull.length; i += 1) {
      alphaHull.push({x: hull[i][0], y: hull[i][1]});
    }

    return alphaHull;
  }

  function _getAlphaHullLib(pointset, concavity, format) {
    let convex,
        concave,
        innerPoints,
        occupiedArea,
        maxSearchArea,
        cellSize,
        points,
        maxEdgeLen = concavity;

    const MAX_CONCAVE_ANGLE_COS = Math.cos(90 / (180 / Math.PI)),
          MAX_SEARCH_BBOX_SIZE_PERCENT = 0.6;

    if (pointset.length < 4) {
      return pointset.slice();
    }

    points = _filterDuplicates(_sortByX(_toXy(pointset, format)));

    occupiedArea = _occupiedArea(points);
    maxSearchArea = [
      occupiedArea[0] * MAX_SEARCH_BBOX_SIZE_PERCENT,
      occupiedArea[1] * MAX_SEARCH_BBOX_SIZE_PERCENT
    ];

    convex = _computeConvexHull(points);
    innerPoints = points.filter(function(pt) {
      return convex.indexOf(pt) < 0;
    });

    cellSize = Math.ceil(1 / (points.length / (occupiedArea[0] * occupiedArea[1])));

    concave = _concave(
      convex, Math.pow(maxEdgeLen, 2),
      maxSearchArea, _grid(innerPoints, cellSize), {});
 
    return _fromXy(concave, format);
  }

  function _filterDuplicates(pointset) {
    return pointset.filter(function(el, idx, arr) {
      let prevEl = arr[idx - 1];
      return idx === 0 || !(prevEl[0] === el[0] && prevEl[1] === el[1]);
    });
  }

  function _sortByX(pointset) {
    return pointset.sort(function(a, b) {
      if (a[0] == b[0]) {
        return a[1] - b[1];
      } else {
        return a[0] - b[0];
      }
    });
  }

  function _toXy(pointset, format) {
    if (format === undefined) {
      return pointset.slice();
    }

    return pointset.map(function(pt) {
      let _getXY = new Function('pt', 'return [pt' + format[0] + ',' + 'pt' + format[1] + '];');
      return _getXY(pt);
    });
  }

  function _fromXy(pointset, format) {
    if (format === undefined) {
      return pointset.slice();
    }

    return pointset.map(function(pt) {
      let _getObj = new Function('pt', 'let o = {}; o' + format[0] + '= pt[0]; o' + format[1] + '= pt[1]; return o;');
      return _getObj(pt);
    });
  }

  function _occupiedArea(pointset) {
    let minX = Infinity, maxX = -Infinity,
        minY = Infinity, maxY = -Infinity;

    for (let i = pointset.length - 1; i >= 0; i -= 1) {
      if (pointset[i][0] < minX) {
        minX = pointset[i][0];
      }
      if (pointset[i][1] < minY) {
        minY = pointset[i][1];
      }
      if (pointset[i][0] > maxX) {
        maxX = pointset[i][0];
      }
      if (pointset[i][1] > maxY) {
        maxY = pointset[i][1];
      }
    }

    return [
      maxX - minX, 
      maxY - minY 
    ];
  }

  function _cross(o, a, b) {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  }

  function _upperTangent(pointset) {
    let lower = [];

    for (let l = 0; l < pointset.length; l++) {
        while (lower.length >= 2 && (_cross(lower[lower.length - 2], lower[lower.length - 1], pointset[l]) <= 0)) {
            lower.pop();
        }
        lower.push(pointset[l]);
    }
    lower.pop();

    return lower;
  }

  function _lowerTangent(pointset) {
    let reversed = pointset.reverse(),
        upper = [];

    for (let u = 0; u < reversed.length; u++) {
        while (upper.length >= 2 && (_cross(upper[upper.length - 2], upper[upper.length - 1], reversed[u]) <= 0)) {
            upper.pop();
        }
        upper.push(reversed[u]);
    }
    upper.pop();

    return upper;
  }

  function _computeConvexHull(pointset) {
    let convex,
        upper = _upperTangent(pointset),
        lower = _lowerTangent(pointset);
    convex = lower.concat(upper);
    convex.push(pointset[0]);  
    return convex;  
  }

  function _concave(convex, maxSqEdgeLen, maxSearchArea, grid, edgeSkipList) {
    let edge,
        keyInSkipList,
        scaleFactor,
        midPoint,
        bBoxAround,
        bBoxWidth,
        bBoxHeight,
        midPointInserted = false;

    for (let i = 0; i < convex.length - 1; i++) {
      edge = [convex[i], convex[i + 1]];
      keyInSkipList = edge[0].join() + ',' + edge[1].join();

      if (_sqLength(edge[0], edge[1]) < maxSqEdgeLen ||
        edgeSkipList[keyInSkipList] === true) { continue; }

      scaleFactor = 0;
      bBoxAround = _bBoxAround(edge);
      do {
        bBoxAround = grid._extendBbox(bBoxAround, scaleFactor);
        bBoxWidth = bBoxAround[2] - bBoxAround[0];
        bBoxHeight = bBoxAround[3] - bBoxAround[1];

        midPoint = _midPoint(edge, grid._rangePoints(bBoxAround), convex);            
        scaleFactor++;
      } while (midPoint === null && (maxSearchArea[0] > bBoxWidth || maxSearchArea[1] > bBoxHeight));

      if (bBoxWidth >= maxSearchArea[0] && bBoxHeight >= maxSearchArea[1]) {
        edgeSkipList[keyInSkipList] = true;
      }

      if (midPoint !== null) {
        convex.splice(i + 1, 0, midPoint);
        grid._removePoint(midPoint);
        midPointInserted = true;
      }
    }

    if (midPointInserted) {
      return _concave(convex, maxSqEdgeLen, maxSearchArea, grid, edgeSkipList);
    }
    return convex;
  }

  function Grid(points, cellSize) {
    this._cells = [];
    this._cellSize = cellSize;

    points.forEach(function(point) {
      let cellXY = this._point2CellXY(point),
          x = cellXY[0],
          y = cellXY[1];
      if (this._cells[x] === undefined) {
        this._cells[x] = [];
      }
      if (this._cells[x][y] === undefined) {
        this._cells[x][y] = [];
      }
      this._cells[x][y].push(point);
    }, this);
  }

  Grid.prototype = {
    _cellPoints: function(x, y) { // (Number, Number) -> Array
      return (this._cells[x] !== undefined && this._cells[x][y] !== undefined) ? this._cells[x][y] : [];
    },

    _rangePoints: function(bbox) { // (Array) -> Array
      let tlCellXY = this._point2CellXY([bbox[0], bbox[1]]),
          brCellXY = this._point2CellXY([bbox[2], bbox[3]]),
          points = [];

      for (let x = tlCellXY[0]; x <= brCellXY[0]; x++) {
        for (let y = tlCellXY[1]; y <= brCellXY[1]; y++) {
          points = points.concat(this._cellPoints(x, y));
        }
      }

      return points;
    },

    _removePoint: function(point) { // (Array) -> Array
      let cellXY = this._point2CellXY(point),
          cell = this._cells[cellXY[0]][cellXY[1]],
          pointIdxInCell;
        
      for (let i = 0; i < cell.length; i++) {
        if (cell[i][0] === point[0] && cell[i][1] === point[1]) {
          pointIdxInCell = i;
          break;
        }
      }

      cell.splice(pointIdxInCell, 1);

      return cell;
    },

    _point2CellXY: function(point) { // (Array) -> Array
      let x = parseInt(point[0] / this._cellSize),
          y = parseInt(point[1] / this._cellSize);
      return [x, y];
    },

    _extendBbox: function(bbox, scaleFactor) { // (Array, Number) -> Array
      return [
        bbox[0] - (scaleFactor * this._cellSize),
        bbox[1] - (scaleFactor * this._cellSize),
        bbox[2] + (scaleFactor * this._cellSize),
        bbox[3] + (scaleFactor * this._cellSize)
      ];
    }
  };

  function _grid(points, cellSize) {
    return new Grid(points, cellSize);
  }

  function _sqLength(a, b) {
    return Math.pow(b[0] - a[0], 2) + Math.pow(b[1] - a[1], 2);
  }

  function _bBoxAround(edge) {
    return [
      Math.min(edge[0][0], edge[1][0]), // left
      Math.min(edge[0][1], edge[1][1]), // top
      Math.max(edge[0][0], edge[1][0]), // right
      Math.max(edge[0][1], edge[1][1])  // bottom
    ];
  }

  function _midPoint(edge, innerPoints, convex) {
    let point = null,
        angle1Cos = Math.cos(90 / (180 / Math.PI)),
        angle2Cos = Math.cos(90 / (180 / Math.PI)),
        a1Cos, a2Cos;

    for (let i = 0; i < innerPoints.length; i++) {
      a1Cos = _cos(edge[0], edge[1], innerPoints[i]);
      a2Cos = _cos(edge[1], edge[0], innerPoints[i]);

      if (a1Cos > angle1Cos && a2Cos > angle2Cos &&
        !_intersect([edge[0], innerPoints[i]], convex) &&
        !_intersect([edge[1], innerPoints[i]], convex)) {

        angle1Cos = a1Cos;
        angle2Cos = a2Cos;
        point = innerPoints[i];
      }
    }

    return point;
  }

  function _cos(o, a, b) {
    let aShifted = [a[0] - o[0], a[1] - o[1]],
        bShifted = [b[0] - o[0], b[1] - o[1]],
        sqALen = _sqLength(o, a),
        sqBLen = _sqLength(o, b),
        dot = aShifted[0] * bShifted[0] + aShifted[1] * bShifted[1];
    return dot / Math.sqrt(sqALen * sqBLen);
  }

  function _intersect(segment, pointset) {
    for (let i = 0; i < pointset.length - 1; i++) {
      let seg = [pointset[i], pointset[i + 1]];
      if (segment[0][0] === seg[0][0] && segment[0][1] === seg[0][1] ||
        segment[0][0] === seg[1][0] && segment[0][1] === seg[1][1]) {
        continue;
      }
      if (intersect(segment, seg)) {
        return true;
      }
    }
    return false;
  }

  function _ccw(x1, y1, x2, y2, x3, y3) {           
    let cw = ((y3 - y1) * (x2 - x1)) - ((y2 - y1) * (x3 - x1));
    return cw > 0 ? true : cw < 0 ? false : true; // colinear
  }

  function intersect(seg1, seg2) {
    let x1 = seg1[0][0], y1 = seg1[0][1],
        x2 = seg1[1][0], y2 = seg1[1][1],
        x3 = seg2[0][0], y3 = seg2[0][1],
        x4 = seg2[1][0], y4 = seg2[1][1];

    return _ccw(x1, y1, x3, y3, x4, y4) !== _ccw(x2, y2, x3, y3, x4, y4) && _ccw(x1, y1, x2, y2, x3, y3) !== _ccw(x1, y1, x2, y2, x4, y4);
  }

  function _getConvexHull(dataset) {
    let pointset = _formatDataset(dataset),
        points,
        hull,
        convexHull = [];

    if (pointset.length < 4) {
      return pointset.slice();
    }
    points = _filterDuplicates(_sortByX(_toXy(pointset)));
    hull = _computeConvexHull(points);

    for(let i = 0; i < hull.length; i += 1) {
      convexHull.push({x: hull[i][0], y: hull[i][1]});
    }
    return convexHull;
  }

  function _getHullPerimeter(hull) {
    let perimeter = 0;
    for(let i = 0; i < hull.length - 1; i += 1) {
      perimeter += _getEdgeLength(hull[i], hull[i + 1]);
    }
    perimeter += _getEdgeLength(hull[hull.length - 1], hull[0]);
    return perimeter;
  }

  function _getGraphDiameter(mst) {
    let pathMatrix = [], shortPathTable = [], 
        innerMatrix = [], innerPath = [],
        t_j, t_k,
        diameter = 0;

    for(let i = 0; i < mst.adjVex.length; i += 1) {
      innerMatrix.push(i);
      innerPath.push(Infinity);
    }
    for(let i = 0; i < mst.adjVex.length; i += 1) {
      pathMatrix.push([...innerMatrix]);
      shortPathTable.push([...innerPath]);
      shortPathTable[i][i] = 0;
    }
    for(let i = 1; i < mst.adjVex.length; i += 1) {
      shortPathTable[i][mst.adjVex[i]] = mst.adjLength[i];
      shortPathTable[mst.adjVex[i]][i] = mst.adjLength[i];
    }

    _shortestPathFloyd(mst, pathMatrix, shortPathTable);

    for(let i = 0; i < mst.adjVex.length; i += 1) {
      for(let j = i + 1; j < mst.adjVex.length; j += 1) {
        if (shortPathTable[i][j] > diameter) {
          diameter = shortPathTable[i][j];
          t_j = i; t_k = j;
        }
      }
    }

    return {
      diameter: diameter,
      t_j: t_j,
      t_k: t_k
    };
  }

  function _shortestPathFloyd(mst, pathMatrix, shortPathTable) {
    let v, w, k;
    for(k = 0; k < mst.adjVex.length; k += 1) {
      for(v = 0; v < mst.adjVex.length; v += 1) {
        for(w = 0; w < mst.adjVex.length; w += 1) {
          if (shortPathTable[v][w] > shortPathTable[v][k] + shortPathTable[k][w]) {
            shortPathTable[v][w] = shortPathTable[v][k] + shortPathTable[k][w];
            pathMatrix[v][w] = pathMatrix[v][k];
          }
        }
      }
    }
  }

  function _getEdgeCos(dataset, mst, index) {
    let a, b,
        d1, d2, d3; 

    if (index !== 0) {
      b = mst.adjVex[index];
    } else {
      for(let i = 1; i < mst.adjVex.length; i += 1) {
        if (mst.adjVex[i] === index) {
          b = i;
          break;
        }
      }
    }

    for(let i = 1; i < mst.adjVex.length; i += 1) {
      if (mst.adjVex[i] === index && i !== b) {
        a = i;
      } else if (i === index && mst.adjVex[i] !== b) {
        a = mst.adjVex[i];
      }
    }

    d1 = _getEdgeLength(dataset[index], dataset[a]);
    d2 = _getEdgeLength(dataset[index], dataset[b]);
    d3 = _getEdgeLength(dataset[a], dataset[b]);
    return (d1 ** 2 + d2 ** 2 - d3 ** 2) / 2 / d1 / d2;
  }

  function _getMaxEdge(index, adjPoint, mst) {
    let maxEdge = 0, adj;

    for(let i = 0; i < adjPoint.length; i += 1) {
      adj = adjPoint[i];
      if (mst.adjVex[index] === adj && mst.adjLength[index] > maxEdge) {
        maxEdge = mst.adjLength[index];
      }
      if (mst.adjVex[adj] === index && mst.adjLength[adj] > maxEdge) {
        maxEdge = mst.adjLength[adj];
      }
    }
    return maxEdge;
  }

/////////////////////////////////////////////////////////////////////////////////
  function getCOutlying(dataset) {
    let mst = _getMSTPrim(_getAdjMatrix(dataset));
        omega = _getOmega(mst);
        degree = [], 
        outliersLength = 0;

    for(let i = 0; i < mst.adjVex.length; i += 1) {
      degree.push(0);
    }
    for(let i = 1; i < mst.adjVex.length; i += 1) {
      degree[i] += 1;
      degree[mst.adjVex[i]] += 1;
    }

    for(let i = 0; i < degree.length; i += 1) {
      if (degree[i] === 1) {
        for(let j = 1; j < mst.adjVex.length; j += 1) {
          if ((j === i || mst.adjVex[j] === i) && mst.adjLength[j] > omega) {
            outliersLength += mst.adjLength[j];
            break;
          }
        }
      }
    }

    return outliersLength / _getGraphLength(mst);
  }

  function getCConvex(dataset) {
    const epsilon = 1e-5;
    const areaOfAlphaHull = _getAHArea(dataset);
    const areaOfConvexHull = _getCHArea(dataset);

    if (areaOfConvexHull === 0 || Math.abs(areaOfAlphaHull - areaOfConvexHull) < epsilon) {
      return 1;
    }

    return areaOfAlphaHull / areaOfConvexHull;
  }

  function getCSkinny(dataset) {
    let mst = _getMSTPrim(_getAdjMatrix(dataset)),
        alpha = _getOmega(mst),
        alphaHull = _getAlphaHull(dataset, alpha),
        area = _getHullArea(alphaHull),
        perimeter = _getHullPerimeter(alphaHull);

    return 1 - Math.sqrt(4 * Math.PI * area) / perimeter;
  }

  function getCStringy(dataset) {
    let mst = _getMSTPrim(_getAdjMatrix(dataset));
    return _getGraphDiameter(mst).diameter / _getGraphLength(mst);
  }

  function getCStraight(dataset, distType = "Euclidean") {
    let mst = _getMSTPrim(_getAdjMatrix(dataset)),
        {diameter, t_j, t_k} = _getGraphDiameter(mst);
    return _getEdgeLength(dataset[t_j], dataset[t_k], distType) / diameter;
  }

  function getCMonotonic(dataset) {
    let {xDataset, yDataset} = _splitDataset(dataset);
        sum_d_i2 = 0;

    for(let i = 0; i < dataset.length; i += 1) {
      xDataset[i] = {val: xDataset[i], index: i};
      yDataset[i] = {val: yDataset[i], index: i};
    }

    xDataset.sort((a, b) => b.val - a.val);
    yDataset.sort((a, b) => b.val - a.val);

    for(let i = 0; i < dataset.length; i += 1) {
      sum_d_i2 += (xDataset[i].index - yDataset[i].index) ** 2;
    }

    return (1 - 6 * sum_d_i2 / dataset.length / (dataset.length ** 2 - 1)) ** 2;
  }

  function getCSkew(dataset) {
    let mst = _getMSTPrim(_getAdjMatrix(dataset)),
        [tmp, ...tmpAdjLength] = mst.adjLength;
    tmpAdjLength.sort((a, b) => a - b);

    let q90 = tmpAdjLength[Math.round(tmpAdjLength.length * 0.9) - 1],
        q50 = tmpAdjLength[Math.round(tmpAdjLength.length * 0.5) - 1],
        q10 = tmpAdjLength[Math.round(tmpAdjLength.length * 0.1) - 1 > 0 ?
          Math.round(tmpAdjLength.length * 0.1) - 1 : 0];

    return (q90 - q50) / (q90 - q10);
  }

  function getCClumpy(dataset) {
    let mst = _getMSTPrim(_getAdjMatrix(dataset)),
        maxCClumpy = 0;
        degree = [];
        
    for(let i = 0; i < mst.adjVex.length; i += 1) {
      degree.push(0);
    }
    for(let i = 1; i < mst.adjVex.length; i += 1) {
      degree[i] += 1;
      degree[mst.adjVex[i]] += 1;
    }

    for(let j = 1; j < mst.adjLength.length; j += 1) {
      if (degree[j] === 1 || degree[mst.adjVex[j]] === 1) continue;

      let a = j, b = mst.adjVex[j];
      let adj_a = [], adj_b = [];
      let jClumpy;

      for(let k = 1; k < mst.adjLength.length; k += 1) {
        if (k === a && mst.adjVex[k] !== b) {
          adj_a.push(mst.adjVex[k]);
        } else if (mst.adjVex[k] === a && k !== b) {
          adj_a.push(k);
        }
        if (k === b && mst.adjVex[k] !== a) {
          adj_b.push(mst.adjVex[k]);
        } else if (mst.adjVex[k] === b && k !== a) {
          adj_b.push(k);
        }
      }

      if (adj_a.length === adj_b.length) {
        jClumpy = 1 - Math.max(_getMaxEdge(a, adj_a, mst), _getMaxEdge(b, adj_b, mst)) / mst.adjLength[j];
      } else if (adj_a.length > adj_b.length) {
        jClumpy = 1 - _getMaxEdge(b, adj_b, mst) / mst.adjLength[j];
      } else {
        jClumpy = 1 - _getMaxEdge(a, adj_a, mst) / mst.adjLength[j];
      }
      maxCClumpy = jClumpy > maxCClumpy ? jClumpy : maxCClumpy;
    }
    return maxCClumpy;
  }

  function getCStriated(dataset) {
    let mst = _getMSTPrim(_getAdjMatrix(dataset)),
        degree = [], 
        V2 = [],
        sumCos = 0;

    for(let i = 0; i < mst.adjVex.length; i += 1) {
      degree.push(0);
    }
    for(let i = 1; i < mst.adjVex.length; i += 1) {
      degree[i] += 1;
      degree[mst.adjVex[i]] += 1;
    }

    for(let i = 0; i < degree.length; i += 1) {
      if (degree[i] === 2) {
        V2.push(i);
      }
    }
    for(let i = 0; i < V2.length; i += 1) {
      sumCos += Math.abs(_getEdgeCos(dataset, mst, V2[i]));
    }
    return sumCos / V2.length;
  }

  return {
    getDomain,
    deduplicateDataset,
    normalizeDataset,
    getCOutlying,
    getCConvex,
    getCSkinny,
    getCStringy,
    getCStraight,
    getCMonotonic,
    getCSkew,
    getCClumpy,
    getCStriated
  }
})()

module.exports = scagnosticIndices;