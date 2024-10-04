#pragma once
#include "../../csv-parser/single_include/csv.hpp"
//  TODO EVERYTHING IS NOT WORKING
namespace arvc{

void write_poses_to_csv(){
    csv::CSVFormat format;
    format.delimiter({'\t', ','})
      .quote('~')
      .header_row(2);


    std::ofstream outfile("poses.csv");
    csv::CSVWriter<std::ofstream> writer(outfile);
    writer << std::vector<std::string>({"x", "y", "z", "qx", "qy", "qz", "qw"});
    writer << std::vector<std::string>({"0.32217392325401306", "0.0", "0.924475371837616", "0.0", "0.0", "0.0", "1.0"});
    writer << std::vector<std::string>({"0.7466145157814026", "0.0", "0.924475371837616", "-0.090690478682518", "-0.103418730199337", "-0.0094695333391428", "0.9904494881629944"});
    writer << std::vector<std::string>({"1.0272488594055176", "0.403963565826416", "0.9074787497520447", "0.09720907360315323", "-0.1582057625055313", "0.14935386180877686", "0.9711924195289612"});
    writer << std::vector<std::string>({"1.2371149063110352", "0.7707521915435791", "1.0632039308547974", "-0.13486790657043457", "0.04330377280712128", "0.08324363082647324", "0.9864106774330139"});
}


void read_poses_from_csv(){
    
    tf2::Transform tmp_tf;

    csv::CSVReader reader("examples/poses.csv");
    int count = 0;
    
    for (csv::CSVRow& row: reader)
    {
        std::cout << "--- POSE " << count++ << std::endl;
        tmp_tf.setOrigin(tf2::Vector3(row["x"].get<double>(), row["y"].get<double>(), row["z"].get<double>()));
        tmp_tf.setRotation(tf2::Quaternion(row["qx"].get<double>(), row["qy"].get<double>(), row["qz"].get<double>(), row["qw"].get<double>()));

        std::cout << tf2::toMsg(tmp_tf) << std::endl;
        
        real_poses.push_back(tmp_tf);    
    }
    
}

}