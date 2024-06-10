From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 83
      | (1, 2, 4) => 81
      | (2, 0, 2) => 19
      | (2, 3, 2) => 45
      | (2, 5, 2) => 26
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 75
      | (1, 2, 4) => 74
      | (2, 0, 2) => 77
      | (2, 3, 2) => 55
      | (2, 5, 2) => 82
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 42
      | (0, 3, 6) => 50
      | (0, 5, 3) => 53
      | (0, 5, 5) => 0
      | (0, 5, 6) => 49
      | (0, 6, 3) => 50
      | (0, 6, 5) => 46
      | (0, 6, 6) => 50
      | _ => 500
      end,
      (bindGen 
      (* GenNat1 *)
      (let _weight_1 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 49
      | (0, 3, 6) => 50
      | (0, 5, 3) => 51
      | (0, 5, 5) => 38
      | (0, 5, 6) => 48
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 50
      | (0, 3, 6) => 50
      | (0, 5, 3) => 52
      | (0, 5, 5) => 46
      | (0, 5, 6) => 48
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 49
      | (0, 3, 6) => 50
      | (0, 5, 3) => 45
      | (0, 5, 5) => 31
      | (0, 5, 6) => 48
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
      (let _weight_8 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 49
      | (0, 3, 6) => 50
      | (0, 5, 3) => 45
      | (0, 5, 5) => 21
      | (0, 5, 6) => 48
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_8, returnGen 8);
        (100-_weight_8, returnGen 0)
      ]) (fun n8 =>
      (let _weight_16 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 49
      | (0, 3, 6) => 50
      | (0, 5, 3) => 45
      | (0, 5, 5) => 21
      | (0, 5, 6) => 48
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_16, returnGen 16);
        (100-_weight_16, returnGen 0)
      ]) (fun n16 =>
      (let _weight_32 := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 49
      | (0, 3, 6) => 50
      | (0, 5, 3) => 45
      | (0, 5, 5) => 21
      | (0, 5, 6) => 48
      | (0, 6, 3) => 50
      | (0, 6, 5) => 50
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_32, returnGen 32);
        (100-_weight_32, returnGen 0)
      ]) (fun n32 =>
        returnGen (n1 + n2 + n4 + n8 + n16 + n32)
      )))))))))))) 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 57
      | (0, 3, 6) => 50
      | (0, 5, 3) => 47
      | (0, 5, 5) => 88
      | (0, 5, 6) => 51
      | (0, 6, 3) => 50
      | (0, 6, 5) => 53
      | (0, 6, 6) => 50
      | _ => 500
      end,
      (bindGen 
      (* GenBool1 *) (let _weight_true := match (size, stack1, stack2) with
      | (0, 3, 3) => 50
      | (0, 3, 5) => 45
      | (0, 3, 6) => 50
      | (0, 5, 3) => 50
      | (0, 5, 5) => 44
      | (0, 5, 6) => 50
      | (0, 6, 3) => 50
      | (0, 6, 5) => 53
      | (0, 6, 6) => 50
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 43
      | (1, 3, 6) => 49
      | (1, 5, 3) => 48
      | (1, 5, 5) => 0
      | (1, 5, 6) => 46
      | (1, 6, 3) => 50
      | (1, 6, 5) => 48
      | (1, 6, 6) => 50
      | (2, 3, 3) => 48
      | (2, 3, 5) => 44
      | (2, 3, 6) => 48
      | (2, 5, 3) => 46
      | (2, 5, 5) => 0
      | (2, 5, 6) => 43
      | (2, 6, 3) => 50
      | (2, 6, 5) => 46
      | (2, 6, 6) => 50
      | (3, 3, 3) => 49
      | (3, 3, 5) => 42
      | (3, 3, 6) => 49
      | (3, 5, 3) => 45
      | (3, 5, 5) => 0
      | (3, 5, 6) => 44
      | (3, 6, 3) => 49
      | (3, 6, 5) => 47
      | (3, 6, 6) => 49
      | (4, 0, 3) => 41
      | (4, 0, 5) => 0
      | (4, 0, 6) => 41
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      (* GenNat2 *)
      (let _weight_1 := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 50
      | (1, 5, 5) => 60
      | (1, 5, 6) => 49
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 49
      | (2, 5, 5) => 52
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 49
      | (3, 5, 5) => 53
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 45
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_1, returnGen 1);
        (100-_weight_1, returnGen 0)
      ]) (fun n1 =>
      (let _weight_2 := match (size, stack1, stack2) with
      | (1, 3, 3) => 49
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 50
      | (1, 5, 5) => 53
      | (1, 5, 6) => 51
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 47
      | (2, 5, 5) => 43
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 49
      | (3, 5, 5) => 41
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 45
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_2, returnGen 2);
        (100-_weight_2, returnGen 0)
      ]) (fun n2 =>
      (let _weight_4 := match (size, stack1, stack2) with
      | (1, 3, 3) => 49
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 47
      | (1, 5, 5) => 9
      | (1, 5, 6) => 49
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 47
      | (2, 5, 5) => 39
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 49
      | (3, 5, 5) => 41
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 45
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_4, returnGen 4);
        (100-_weight_4, returnGen 0)
      ]) (fun n4 =>
      (let _weight_8 := match (size, stack1, stack2) with
      | (1, 3, 3) => 49
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 47
      | (1, 5, 5) => 9
      | (1, 5, 6) => 49
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 47
      | (2, 5, 5) => 39
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 49
      | (3, 5, 5) => 41
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 45
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_8, returnGen 8);
        (100-_weight_8, returnGen 0)
      ]) (fun n8 =>
      (let _weight_16 := match (size, stack1, stack2) with
      | (1, 3, 3) => 49
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 47
      | (1, 5, 5) => 9
      | (1, 5, 6) => 49
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 47
      | (2, 5, 5) => 39
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 49
      | (3, 5, 5) => 41
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 45
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_16, returnGen 16);
        (100-_weight_16, returnGen 0)
      ]) (fun n16 =>
      (let _weight_32 := match (size, stack1, stack2) with
      | (1, 3, 3) => 49
      | (1, 3, 5) => 50
      | (1, 3, 6) => 50
      | (1, 5, 3) => 47
      | (1, 5, 5) => 9
      | (1, 5, 6) => 49
      | (1, 6, 3) => 50
      | (1, 6, 5) => 50
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 50
      | (2, 3, 6) => 50
      | (2, 5, 3) => 47
      | (2, 5, 5) => 39
      | (2, 5, 6) => 50
      | (2, 6, 3) => 50
      | (2, 6, 5) => 50
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 50
      | (3, 5, 3) => 49
      | (3, 5, 5) => 41
      | (3, 5, 6) => 50
      | (3, 6, 3) => 50
      | (3, 6, 5) => 50
      | (3, 6, 6) => 50
      | (4, 0, 3) => 50
      | (4, 0, 5) => 45
      | (4, 0, 6) => 50
      | (5, 0, 0) => 50
      | _ => 500
      end
      in
      bindGen (freq [
        (_weight_32, returnGen 32);
        (100-_weight_32, returnGen 0)
      ]) (fun n32 =>
        returnGen (n1 + n2 + n4 + n8 + n16 + n32)
      )))))))))))) 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 49
      | (1, 3, 5) => 59
      | (1, 3, 6) => 51
      | (1, 5, 3) => 45
      | (1, 5, 5) => 16
      | (1, 5, 6) => 58
      | (1, 6, 3) => 50
      | (1, 6, 5) => 55
      | (1, 6, 6) => 50
      | (2, 3, 3) => 48
      | (2, 3, 5) => 56
      | (2, 3, 6) => 55
      | (2, 5, 3) => 43
      | (2, 5, 5) => 3
      | (2, 5, 6) => 61
      | (2, 6, 3) => 50
      | (2, 6, 5) => 54
      | (2, 6, 6) => 50
      | (3, 3, 3) => 49
      | (3, 3, 5) => 60
      | (3, 3, 6) => 50
      | (3, 5, 3) => 44
      | (3, 5, 5) => 1
      | (3, 5, 6) => 55
      | (3, 6, 3) => 49
      | (3, 6, 5) => 54
      | (3, 6, 6) => 51
      | (4, 0, 3) => 41
      | (4, 0, 5) => 0
      | (4, 0, 6) => 61
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      (* GenBool2 *) (let _weight_true := match (size, stack1, stack2) with
      | (1, 3, 3) => 50
      | (1, 3, 5) => 47
      | (1, 3, 6) => 49
      | (1, 5, 3) => 50
      | (1, 5, 5) => 47
      | (1, 5, 6) => 53
      | (1, 6, 3) => 50
      | (1, 6, 5) => 52
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 56
      | (2, 3, 6) => 49
      | (2, 5, 3) => 50
      | (2, 5, 5) => 45
      | (2, 5, 6) => 55
      | (2, 6, 3) => 50
      | (2, 6, 5) => 49
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 50
      | (3, 3, 6) => 51
      | (3, 5, 3) => 50
      | (3, 5, 5) => 48
      | (3, 5, 6) => 51
      | (3, 6, 3) => 50
      | (3, 6, 5) => 51
      | (3, 6, 6) => 52
      | (4, 0, 3) => 50
      | (4, 0, 5) => 42
      | (4, 0, 6) => 48
      | (5, 0, 0) => 56
      | _ => 500
      end
      in
      freq [
        (_weight_true, returnGen true);
        (100-_weight_true, returnGen false)
      ]) 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 52
      | (1, 3, 5) => 53
      | (1, 3, 6) => 51
      | (1, 5, 3) => 60
      | (1, 5, 5) => 95
      | (1, 5, 6) => 50
      | (1, 6, 3) => 50
      | (1, 6, 5) => 49
      | (1, 6, 6) => 50
      | (2, 3, 3) => 54
      | (2, 3, 5) => 53
      | (2, 3, 6) => 48
      | (2, 5, 3) => 62
      | (2, 5, 5) => 96
      | (2, 5, 6) => 51
      | (2, 6, 3) => 51
      | (2, 6, 5) => 53
      | (2, 6, 6) => 51
      | (3, 3, 3) => 53
      | (3, 3, 5) => 53
      | (3, 3, 6) => 52
      | (3, 5, 3) => 58
      | (3, 5, 5) => 96
      | (3, 5, 6) => 54
      | (3, 6, 3) => 52
      | (3, 6, 5) => 51
      | (3, 6, 6) => 50
      | (4, 0, 3) => 67
      | (4, 0, 5) => 96
      | (4, 0, 6) => 52
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 49
      | (1, 3, 5) => 44
      | (1, 3, 6) => 49
      | (1, 5, 3) => 45
      | (1, 5, 5) => 0
      | (1, 5, 6) => 45
      | (1, 6, 3) => 50
      | (1, 6, 5) => 48
      | (1, 6, 6) => 50
      | (2, 3, 3) => 50
      | (2, 3, 5) => 45
      | (2, 3, 6) => 48
      | (2, 5, 3) => 46
      | (2, 5, 5) => 0
      | (2, 5, 6) => 43
      | (2, 6, 3) => 50
      | (2, 6, 5) => 46
      | (2, 6, 6) => 50
      | (3, 3, 3) => 50
      | (3, 3, 5) => 42
      | (3, 3, 6) => 49
      | (3, 5, 3) => 51
      | (3, 5, 5) => 0
      | (3, 5, 6) => 45
      | (3, 6, 3) => 49
      | (3, 6, 5) => 47
      | (3, 6, 6) => 49
      | (4, 0, 3) => 46
      | (4, 0, 5) => 0
      | (4, 0, 6) => 44
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
