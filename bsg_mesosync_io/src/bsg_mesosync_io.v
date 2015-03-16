// Mesosynchronous IO that includes the config_tag converter and
// mesosynchronous channel. This IO module receives configuration
// in config tag protocol, and has the same handshake protocol on 
// IO and chip side as mesosynchronous channel: No handshake on the
// IO side, valid from channel to chip (to know when to sample) and
// ready from chip to channel (to know when to send data).
//
// Most important feature of this IO is least latency.

//`ifndef DEFINITIONS_V
//`include "definitions.v"
//`endif

module bsg_mesosync_io
                  #(  parameter ch1_width_p = -1
                    , parameter ch2_width_p = -1
                    , parameter log_LA_fifo_depth_p = -1
                    , parameter cfg_tag_base_id_p = -1
                    
                    , parameter width_lp = ch1_width_p + ch2_width_p
                   )
                   (  input                          clk
                    , input                          reset
                    
                    , input  config_s                config_i

                    // Sinals with their acknowledge
                    , input  [width_lp-1:0]          IO_i

                    , output logic [width_lp-1:0]    chip_o
                    , output                         valid_o
                    
                    , input  [width_lp-1:0]          chip_i
                    , output                         ready_o

                    , output [width_lp-1:0]          IO_o

                    , output logic                   loop_back_o
                    );

// internal singals
clk_divider_s                clk_divider;
bit_cfg_s [width_lp-1:0]     bit_cfg;
mode_cfg_s                   mode_cfg;
logic [$clog2(width_lp)-1:0] input_bit_selector_ch1;
logic [$clog2(width_lp)-1:0] output_bit_selector_ch1;
logic [$clog2(width_lp)-1:0] input_bit_selector_ch2;
logic [$clog2(width_lp)-1:0] output_bit_selector_ch2;
logic                        loop_back;
logic                        channel_reset;
logic                        ready, valid;
logic [width_lp-1:0]         to_IO, to_chip;
 
// config tag extractor module that extracts the configurations from
// config tag serial input and gives them to mesosynchronous channel
bsg_mesosync_config_tag_extractor
           #(  .ch1_width_p(ch1_width_p)
             , .ch2_width_p(ch2_width_p)
             , .cfg_tag_base_id_p(cfg_tag_base_id_p)
            ) cnfg_tag_extractor
            (  .clk(clk)
             , .reset(reset)
             
             , .config_i(config_i)

             // Configuration output
             , .clk_divider_o(clk_divider)
             , .bit_cfg_o(bit_cfg)
             , .mode_cfg_o(mode_cfg)
             , .input_bit_selector_ch1_o(input_bit_selector_ch1)
             , .output_bit_selector_ch1_o(output_bit_selector_ch1)
             , .input_bit_selector_ch2_o(input_bit_selector_ch2)
             , .output_bit_selector_ch2_o(output_bit_selector_ch2)
             , .loop_back_o(loop_back)
             , .channel_reset_o(channel_reset)
             );
 
// Mesosynchronous channel
bsg_mesosync_channel 
           #(  .width_p(width_lp)
             , .log_LA_fifo_depth_p(log_LA_fifo_depth_p)
             ) mesosync_channel
            (  .clk(clk)
             , .reset(channel_reset)
             
             // Configuration inputs
             , .clk_divider_i(clk_divider)
             , .bit_cfg_i(bit_cfg)
             , .mode_cfg_i(mode_cfg)
             , .input_bit_selector_ch1_i(input_bit_selector_ch1)
             , .output_bit_selector_ch1_i(output_bit_selector_ch1)
             , .input_bit_selector_ch2_i(input_bit_selector_ch2)
             , .output_bit_selector_ch2_i(output_bit_selector_ch2)

        
             // Sinals with their acknowledge
             , .IO_i(IO_i)

             , .chip_o(to_chip)
             , .valid_o(valid)
             
             , .chip_i(chip_i)
             , .ready_o(ready)

             , .IO_o(to_IO)

             );

// mesosync channel uses the channel reset from config_tag
// hence during reset output of module must be made zero on 
// this top module
always_comb
  if (reset) begin
    IO_o        = 0;
    ready_o     = 0;
    chip_o      = 0;
    valid_o     = 0;
  end else begin 
    IO_o        = to_IO;
    ready_o     = ready; 
    chip_o      = to_chip;
    valid_o     = valid;
  end

 endmodule
