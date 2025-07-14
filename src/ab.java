import java.util.ArrayList;

public class ab{
    static class edge{
        int src;
        int dest;
        int wt;

        public edge(int s,int d,int w){
            this.src=s;
            this.dest=d;
            this.wt=w;
        }
    }
    public static void create(ArrayList<edge> g[]){
        for(int i=0;i<g.length;i++){
            g[i]=new ArrayList<edge>();
        }
        g[0].add(new edge(0,2,2));

        g[1].add(new edge(1,2,10));
        g[1].add(new edge(1,3,0));

        g[2].add(new edge(2,0,2));
        g[2].add(new edge(2,1,10));
        g[2].add(new edge(2,3,-1));

        g[3].add(new edge(3,1,0));
        g[3].add(new edge(3,2,-1));
    }
    public static void main(String[] args) {
        int V=5;
        ArrayList<edge> graph[]=new ArrayList[V];
        create(graph);

        //printing neighbour of 2
        for(int i=0;i<graph[2].size();i++){
            edge e=graph[2].get(i);
            System.out.println(e.dest+" , "+e.wt);
        }

    }
}