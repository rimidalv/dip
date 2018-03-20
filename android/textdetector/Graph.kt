package ru.avangard.tensorflow.textdetector

/**
 * Created by rimidalv on 3/10/18.
 */
class Graph(val graph: List<Array<Boolean>>) {
    fun subGraphsConnected(): ArrayList<ArrayList<Int>> {

        val subGraphs = arrayListOf<ArrayList<Int>>()
        for (i in graph.indices){
            val anyLine = graph[i].any { it == true }
            val anyRow = !graph.map { it[i] }.any { it == true }
            if (anyLine && anyRow){
                val subRow = arrayListOf<Int>()
                var element = i //graph[i].indexOfFirst { it }
                subRow.add(element)
                while (element != -1 && graph[element].any()){
                    element = graph[element].indexOfFirst{ it }
                    if(element != -1)
                        subRow.add(element)
                }
                subGraphs.add(subRow)
            }
        }
        return subGraphs
    }
}