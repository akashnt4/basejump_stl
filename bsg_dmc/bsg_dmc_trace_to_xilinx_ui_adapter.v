///////////////////////////////////////////////////////////////////////////////////////////////////
//    		BASEJUMP STL
//
//       MODULE: bsg_dmc_trace_to_xilinx_ui_adapter
//  DESCRIPTION: Takes the DMC cmd, addr, wdata, wmask trace packet and converts into XILINX UI interface to feed to DMC;
//  			 And converts UI read interface signals to trace packet to forward to FPGA
//    AUTHOR(S): Akash Suresh, akashs3@uw.edu
// ORGANIZATION: Bespoke Silicon Group, University of Washington
//      CREATED: 01/07/22
///////////////////////////////////////////////////////////////////////////////////////////////////

module bsg_dmc_trace_to_xilinx_ui_adapter
	import bsg_dmc_pkg::*;
	#(	parameter `BSG_INV_PARAM( data_width_p),
		parameter `BSG_INV_PARAM( addr_width_p),
		parameter `BSG_INV_PARAM( cmd_width_p),
		parameter `BSG_INV_PARAM( burst_width_p),

		localparam payload_width_lp = data_width_p + (data_width_p>>3) + 4,
		localparam mask_width_lp = data_width_p>>3
	)
	( 	input 									core_clk_i,
		input									core_reset_i,

		input [payload_width_lp -1 :0] 			trace_data_i,
		input 									trace_data_valid_i,

		output logic [payload_width_lp -1 :0]	read_data_to_consumer_o,
		output logic							read_data_to_consumer_valid_o,

		output      							ready_o,
   		// xilinx user interface
   		output logic [addr_width_p-1:0]        	app_addr_o,
   		output app_cmd_e				    	app_cmd_o,
   		output logic                       		app_en_o,
   		input                              		app_rdy_i,
   		output logic                       		app_wdf_wren_o,
   		output logic [data_width_p-1:0]			app_wdf_data_o,
   		output logic [(data_width_p>>3)-1:0] 	app_wdf_mask_o,
   		output logic                   			app_wdf_end_o,
   		input                              		app_wdf_rdy_i,
   		input                              		app_rd_data_valid_i,
   		input [data_width_p-1:0] 		      	app_rd_data_i,
   		input                              		app_rd_data_end_i
	);

	import bsg_dmc_pkg::*;

	`declare_dmc_trace_entry_s(addr_width_p, burst_width_p, data_width_p)

	// counter to load one packet per burst per cycle onto app_wdata and app_wmask
	logic [$clog2(burst_width_p) - 1:0] write_count;

	logic transaction_in_progress;

	logic burst_done;

	dmc_trace_entry_s trace_data;

	assign trace_data = trace_data_i;

	assign ready_o =  ~transaction_in_progress & app_rdy_i ;

	assign burst_done =  (write_count == burst_width_p  - 1);

	assign is_write = (trace_data_valid_i && trace_data.wdata_cmd_n) ? ((trace_data.cmd == WP) || (trace_data.cmd == WR) )  : 0;

    /*
    This module is at a level above the dmc_controller where we stall transactions. So this module issues transactions to the dmc_controller which decides whether to forward it to DRAM or not.
    So this transaction_in_progress is between the adapter and dmc_controller.
    For the case of clock change happening right when a read is in progress, the time for tag value to reflect on the chip side will sufficient to cover the previous read/write transaction.
    */
	assign transaction_in_progress = app_wdf_wren_o | app_rd_data_valid_i; 

	assign app_wdf_end_o = burst_done & app_wdf_wren_o;

	assign {feed_to_app_wdata, feed_to_app_wmask} = trace_data_i;
	
	// counting write_count per burst
	bsg_counter_clear_up
					#(.max_val_p(burst_width_p - 1)
					,.init_val_p(0)
                    ,.disable_overflow_warning_p(1))
					write_counter
					(.clk_i(core_clk_i)
					,.reset_i(core_reset_i)
					,.clear_i(1'b0)                        
					,.up_i(app_wdf_wren_o & app_wdf_rdy_i )
					,.count_o(write_count)
					);
	
	// Convert UI command and addr
	assign app_en_o = (trace_data_valid_i &   app_rdy_i) ? (trace_data.wdata_cmd_n) : 0;
	assign app_cmd_o = trace_data.cmd;
	assign app_addr_o = trace_data.addr;

    always @(posedge core_clk_i) begin
        if(core_reset_i) begin
            app_wdf_wren_o <= 0;
        end
        else if(is_write && app_wdf_rdy_i & trace_data_valid_i) begin
            app_wdf_wren_o <= 1;
        end
        else if(burst_done) begin
            app_wdf_wren_o <= 0;
        end
    end

    assign app_wdf_data_o = ( trace_data_valid_i & ~trace_data.wdata_cmd_n) ? trace_data_i[35:4] : 0;
    assign app_wdf_mask_o = ( trace_data_valid_i & ~trace_data.wdata_cmd_n) ? trace_data_i[3:0] : 0;

	assign read_data_to_consumer_o = (app_rd_data_valid_i) ? app_rd_data_i : 0;
	assign read_data_to_consumer_valid_o = app_rd_data_valid_i ? 1 : 0;
endmodule
